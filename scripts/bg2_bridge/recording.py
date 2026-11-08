"""Per-request inputs/outputs and lossless, timestamped camera-stream recording."""
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image

from .core import require


class RecordingError(RuntimeError):
    pass


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_inputs(folder, observation, raw_frames=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    for key in ['head','wrist_left','wrist']:
        Image.fromarray(observation[key]).save(folder/(key+'.png'))
    composite=np.concatenate([observation['head'],np.concatenate([observation['wrist_left'],observation['wrist']],axis=1)],axis=0)
    Image.fromarray(composite).save(folder/'input_composite.png')
    with (folder/'observation.npz').open('xb') as f:np.savez_compressed(f,**observation)
    meta={'source_time_ns':int(observation['source_time_ns']),'instruction':str(observation['instruction']),
          'joint_names':np.asarray(observation['joint_names']).tolist(),
          'joint_position':np.asarray(observation['joint_position']).tolist(),'units':['rad']*8,'raw_cameras':{}}
    for key,frame in (raw_frames or {}).items():
        source,receipt,payload=frame
        with Image.open(BytesIO(payload)) as image:
            extension={'JPEG':'jpg','PNG':'png'}.get(image.format)
            require(extension is not None,'Unsupported compressed camera format')
            wh=list(image.size)
        name='raw_'+key+'.'+extension
        with (folder/name).open('xb') as f:f.write(payload)
        meta['raw_cameras'][key]={'file':name,'source_time_ns':source,'received_monotonic_ns':receipt,'raw_wh':wh}
    write_json(folder/'observation.json',meta)


class CameraVideo:
    """Remux original JPEG packets; preserve exact sensor stamps in the sidecar."""
    def __init__(self,root,key):
        self.root,self.key=Path(root),key
        self.container,self.stream,self.origin,self.last,self.shape,self.format=None,None,None,None,None,None
        self.count=0
        self.index=(self.root/(key+'.timestamps.jsonl')).open('x')

    def write(self,source,receipt,payload):
        import av
        with Image.open(BytesIO(payload)) as im:kind,shape=im.format,im.size
        require(kind=='JPEG','Continuous camera recording requires JPEG packets')
        require(self.last is None or source>self.last,'Camera recording timestamp did not increase')
        if self.container is None:
            self.origin,self.shape,self.format=source,shape,kind
            self.container=av.open(str(self.root/(self.key+'.mkv')),'w',format='matroska')
            self.stream=self.container.add_stream('mjpeg')
            self.stream.width,self.stream.height=shape
            self.stream.pix_fmt='yuvj420p'
            self.stream.time_base=Fraction(1,1_000_000_000)
        require(shape==self.shape and kind==self.format,'Camera format or dimensions changed during recording')
        packet=av.Packet(payload);packet.stream=self.stream
        packet.pts=packet.dts=source-self.origin;packet.time_base=Fraction(1,1_000_000_000)
        self.container.mux(packet)
        self.index.write(json.dumps({'frame':self.count,'source_time_ns':source,'received_monotonic_ns':receipt,
                                     'relative_time_ns':source-self.origin})+'\n');self.index.flush()
        self.last=source;self.count+=1

    def close(self):
        try:
            if self.container is not None:self.container.close()
        finally:self.index.close()


class RunRecorder:
    def __init__(self,root,mode):
        self.root,self.mode=Path(root),mode
        (self.root/'cameras').mkdir(exist_ok=False)
        (self.root/'predictions').mkdir(exist_ok=False)
        self.videos={key:CameraVideo(self.root/'cameras',key) for key in ['head','wrist_left','wrist']}
        self.error,self.closed=None,False
        self.lock=RLock()
        write_json(self.root/'run.json',{'mode':mode,'physical_publishing':mode=='execute',
                   'policy_mode':'policy','imagined_future_rollout':False,
                   'camera_recording':'original compressed packets in timestamped Matroska; no frame resampling',
                   'model_visual_location':'PRO 6000 server output; linked by epoch/request_id'})

    def check(self):
        if self.error is not None:raise RecordingError(self.error)

    def camera(self,key,source,receipt,payload):
        with self.lock:
            if self.closed:return
            self.check()
            try:self.videos[key].write(source,receipt,payload)
            except Exception as error:
                self.error='Camera recording failed: '+str(error)
                raise RecordingError(self.error) from error

    def inputs(self,request_id,observation,raw_frames):
        self.check()
        try:save_inputs(self.root/'predictions'/request_id,observation,raw_frames)
        except Exception as error:
            self.error='Input recording failed: '+str(error);raise RecordingError(self.error) from error

    def prediction(self,result):
        self.check()
        try:write_json(self.root/'predictions'/result['request_id']/'prediction.json',result)
        except Exception as error:
            self.error='Prediction recording failed: '+str(error);raise RecordingError(self.error) from error

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            errors=[]
            for video in self.videos.values():
                try:video.close()
                except Exception as error:errors.append(str(error))
            write_json(self.root/'recording_summary.json',{'frames':{k:v.count for k,v in self.videos.items()},
                       'error':self.error,'close_errors':errors})
            if errors:raise RecordingError('; '.join(errors))


def save_model_result(root,observation,result,response,sampling,*,request_group='requests'):
    folder=Path(root)/request_group/response['epoch']/response['request_id']
    folder.mkdir(parents=True,exist_ok=False)
    save_inputs(folder/'inputs',observation)
    write_json(folder/'prediction.json',response)
    if 'normalized' in result:
        with (folder/'normalized_actions.npy').open('xb') as f:np.save(f,result['normalized'],allow_pickle=False)
    mode=result.get('mode','policy')
    if mode=='wam':
        from bg2_wam import save_rollout
        with (folder/'raw_joint_positions.npy').open('xb') as f:np.save(f,result['positions'],allow_pickle=False)
        with (folder/'joint_positions.npy').open('xb') as f:np.save(f,np.asarray(response['positions']),allow_pickle=False)
        rollout=save_rollout(folder,result['imagined_video'])
    visuals=[]
    for i,latent in enumerate(result.get('vision',[])):
        stem=f'vision_{i:03d}'
        with (folder/(stem+'_latent.npy')).open('xb') as f:np.save(f,latent,allow_pickle=False)
        entry={'latent':stem+'_latent.npy','latent_shape':list(latent.shape),
               'kind':'generated_wam_rollout_latent' if mode=='wam' else 'conditioned_context_latent',
               'decoded_frames':17 if mode=='wam' else 0}
        visuals.append(entry)
    write_json(folder/'model_output.json',{'sampling':sampling,'prompt':result.get('prompt'),
               'visuals':visuals,'mode':mode,'imagined_future_rollout':mode=='wam',
               'runtime_metadata':result.get('runtime_metadata'),
               'note':('Joint WAM sample; raw actions retained, RPC commands floor negative right-gripper values at zero.' if mode=='wam' else
                   'Policy mode returns the conditioned visual latent; no future-video generation or VAE decoding is requested.')})
    artifacts={'directory':folder.relative_to(root).as_posix(),'visuals':visuals}
    if mode=='wam':
        artifacts.update({'mode':'wam','rollout':rollout,'videos':{name:name+'.mp4' for name in
            ['imagined_future','rollout_full','imagined_head','imagined_wrist_left','imagined_wrist_right']}})
    return artifacts
