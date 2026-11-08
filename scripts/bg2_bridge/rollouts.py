"""Recording-only downloads and per-run imagined previews; never requests predictions."""
from fractions import Fraction
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from .artifact_protocol import ARTIFACT_API, PUBLIC_FILES, artifact_path, atomic_json, run_name


def render_chunk(folder, cancelled=lambda:False):
    """Encode saved RGB forecasts after control has ended; works in the existing gateway image."""
    import av
    folder=Path(folder)
    frames=np.load(folder/'decoded_rollout.npy',mmap_mode='r',allow_pickle=False)
    if frames.shape!=(17,640,512,3) or frames.dtype!=np.uint8:
        raise ValueError('Invalid saved imagined RGB chunk')
    future=frames[1:]
    views={'imagined_future.mp4':future,'imagined_head.mp4':future[:,:384],
           'imagined_wrist_left.mp4':future[:,384:,:256],'imagined_wrist_right.mp4':future[:,384:,256:],
           'rollout_full.mp4':frames}
    for name,pixels in views.items():
        destination=folder/name
        if destination.exists():continue
        temporary=folder/('.'+name+'.tmp.mp4')
        try:
            with av.open(str(temporary),'w',format='mp4',options={'movflags':'+faststart'}) as output:
                stream=output.add_stream('libx264',rate=20,options={'crf':'18','preset':'fast'})
                stream.width,stream.height=pixels.shape[2],pixels.shape[1]
                stream.pix_fmt='yuv420p';stream.codec_context.thread_count=1
                for index,image in enumerate(pixels):
                    if cancelled():raise InterruptedError('Rendering interrupted; raw chunks retained for collection later')
                    frame=av.VideoFrame.from_ndarray(np.ascontiguousarray(image),format='rgb24')
                    frame.pts,frame.time_base=index,Fraction(1,20)
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            temporary.replace(destination)
        finally:
            if temporary.exists():temporary.unlink()


def concatenate(clips, destination):
    """Remux forecast clips in request order without encoding or changing frame rate."""
    import av
    destination=Path(destination)
    temporary=destination.with_name('.'+destination.name+'.tmp.mp4')
    with av.open(str(clips[0])) as first:
        template=first.streams.video[0]
        codec,width,height=template.codec_context.name,template.width,template.height
        with av.open(str(temporary),'w',format='mp4',options={'movflags':'+faststart'}) as output:
            stream=output.add_stream_from_template(template)
            stream.time_base=template.time_base
            offset=Fraction(0)
            for path in clips:
                with av.open(str(path)) as source:
                    video=source.streams.video[0]
                    if (video.codec_context.name,video.width,video.height)!=(codec,width,height):
                        raise ValueError('Imagined clip encoding/dimensions changed inside the run')
                    origin=video.start_time or 0
                    end=Fraction(0)
                    for packet in source.demux(video):
                        if packet.dts is None:continue
                        pts=Fraction(packet.pts-origin)*packet.time_base
                        end=max(end,pts+Fraction(packet.duration or 1)*packet.time_base)
                        shift=int(offset/packet.time_base)-origin
                        packet.pts+=shift;packet.dts+=shift;packet.stream=stream
                        output.mux(packet)
                    offset+=end
    temporary.replace(destination)


class RunRollouts:
    def __init__(self, root, client, *, resume=False):
        self.root=Path(root);self.run_id=run_name(self.root.name)
        self.directory=self.root/'imagined';self.directory.mkdir(exist_ok=resume)
        self.owner=(self.directory/'.collector.lock').open('a')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise ValueError('This run already has an active recording collector')
        self.path=self.directory/'manifest.json'
        self.client=client
        self.lock=threading.RLock();self.stop_event=threading.Event();self.changed=threading.Event()
        self.entries=json.loads(self.path.read_text())['chunks'] if resume and self.path.exists() else []
        self.last_error=None
        self.abort_assembly=False
        if resume and not self.entries:
            previous=[]
            for folder in (self.root/'predictions').glob('*'):
                prediction=folder/'prediction.json';observation=folder/'observation.json'
                if prediction.is_file():
                    data=json.loads(prediction.read_text())
                    if data.get('mode')!='wam':continue
                    previous.append((data['source_time_ns'],data['request_id'],data['epoch'],True))
                elif observation.is_file():
                    data=json.loads(observation.read_text())
                    previous.append((data['source_time_ns'],folder.name,None,False))
            from .rpc import identifier
            for source,ident,epoch,received in sorted(previous):
                identifier(ident)
                self.entries.append({'index':len(self.entries)+1,'request_id':ident,'epoch':epoch,
                    'source_time_ns':source,'prediction_received':received,'state':'pending',
                    'status_path':artifact_path(self.run_id,epoch,ident) if epoch else None,
                    'directory':f'chunks/{len(self.entries)+1:06d}_{ident}'})
        if resume:
            for item in self.entries:
                if item['state']=='failed':item['state']='pending'
        self.save()
        self.thread=threading.Thread(target=self.work,name='imagined-video-downloads',daemon=True)
        self.thread.start()

    def save(self):
        atomic_json(self.path,{'schema':ARTIFACT_API,'run_id':self.run_id,'chunks':self.entries,
            'note':'Forecast chunks in request order; inference/hold gaps are not real-time camera footage.',
            'last_error':self.last_error})

    def register(self, epoch, request_id, source_time_ns):
        with self.lock:
            if any(e['request_id']==request_id for e in self.entries):return
            index=len(self.entries)+1
            self.entries.append({'index':index,'epoch':epoch,'request_id':request_id,
                'source_time_ns':int(source_time_ns),'state':'pending','prediction_received':False,
                'status_path':artifact_path(self.run_id,epoch,request_id),
                'directory':f'chunks/{index:06d}_{request_id}'})
            self.save()
        self.changed.set()

    def prediction(self, result):
        self.register(result['epoch'],result['request_id'],result['source_time_ns'])
        with self.lock:
            entry=next(e for e in self.entries if e['request_id']==result['request_id'])
            entry['prediction_received']=True
            reference=result.get('artifacts',{})
            if reference.get('state')=='failed':entry.update(state='failed',error=reference.get('error'))
            self.save()

    def work(self):
        while not self.stop_event.is_set():
            with self.lock:pending=[dict(e) for e in self.entries if e['state'] not in ('complete','failed')]
            for item in pending:
                if self.stop_event.is_set():break
                try:
                    if item.get('status_path') is None:
                        found=self.client.recording_status('/artifact-lookup/'+self.run_id+'/'+item['request_id'])
                        if found['source_time_ns']!=item['source_time_ns']:raise ValueError('Saved observation timestamp mismatch')
                        item.update(epoch=found['epoch'],status_path=found['status_path'])
                        with self.lock:
                            entry=next(e for e in self.entries if e['request_id']==item['request_id'])
                            entry.update(epoch=item['epoch'],status_path=item['status_path']);self.save()
                    status=self.client.recording_status(item['status_path'])
                    with self.lock:self.last_error=None
                    if (status.get('run_id'),status.get('epoch'),status.get('request_id'))!=(self.run_id,item['epoch'],item['request_id']):
                        raise ValueError('Recording response identity mismatch')
                    if status['state']=='complete':
                        folder=self.directory/item['directory'];folder.mkdir(parents=True,exist_ok=True)
                        for name,expected in status['files'].items():
                            if name not in PUBLIC_FILES:raise ValueError('Unknown recording filename')
                            destination=folder/name
                            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==expected['sha256']:
                                continue
                            self.client.download(item['status_path']+'/'+name,destination,expected)
                    with self.lock:
                        entry=next(e for e in self.entries if e['request_id']==item['request_id'])
                        entry['state']=status['state']
                        if status.get('error'):entry['error']=status['error']
                        self.save()
                except Exception as error:
                    if not self.stop_event.is_set():
                        with self.lock:self.last_error=str(error);self.save()
            self.changed.wait(.5);self.changed.clear()

    def pending(self):
        with self.lock:return sum(e['state'] not in ('complete','failed') for e in self.entries)

    def cancel(self):
        self.abort_assembly=True
        self.stop_event.set();self.client.cancel();self.changed.set()

    def close(self, wait=30):
        deadline=time.monotonic()+wait;last_report=0.
        while self.pending() and not self.stop_event.is_set() and time.monotonic()<deadline:
            if time.monotonic()-last_report>=5:
                print(json.dumps({'event':'collecting_recordings','pending':self.pending(),
                    'note':'Control loop has ended. Ctrl+C again skips this wait; collection can resume later.'}),flush=True)
                last_report=time.monotonic()
            self.changed.set();time.sleep(.05)
        self.stop_event.set();self.client.cancel();self.changed.set();self.thread.join(timeout=2)
        with self.lock:
            complete=[dict(entry) for entry in self.entries if entry['state']=='complete']
            missing=[dict(entry) for entry in self.entries if entry['state']!='complete']
        if complete:
            try:
                print(json.dumps({'event':'rendering_recordings','chunks':len(complete),
                    'note':'Control has ended. Rendering saved chunks and combined previews; Ctrl+C again defers rendering.'}),flush=True)
                for item in complete:
                    if self.abort_assembly:raise InterruptedError('Rendering deferred; raw recordings retained')
                    folder=self.directory/item['directory']
                    if (folder/'decoded_rollout.npy').exists():render_chunk(folder,lambda:self.abort_assembly)
                    with self.lock:
                        next(e for e in self.entries if e['request_id']==item['request_id'])['rendered']=True
                for source,target in [('imagined_future.mp4','rollout.mp4'),('imagined_head.mp4','head.mp4'),
                        ('imagined_wrist_left.mp4','wrist_left.mp4'),('imagined_wrist_right.mp4','wrist_right.mp4')]:
                    concatenate([self.directory/e['directory']/source for e in complete],self.directory/target)
                atomic_json(self.directory/'timeline.json',{'fps':20,'chunks':[
                    {**e,'first_frame':index*16,'last_frame':index*16+15} for index,e in enumerate(complete)],
                    'frame_count':len(complete)*16,'pending':self.pending(),
                    'missing_chunks':missing,'partial':bool(missing),
                    'note':'Concatenated imagined forecasts, not continuous physical time.'})
                run_meta=self.root/'run.json'
                if run_meta.exists():
                    data=json.loads(run_meta.read_text())
                    atomic_json(run_meta,{**data,'policy_mode':'wam','imagined_future_rollout':True,
                                          'model_visual_location':'imagined/'})
            except Exception as error:
                self.last_error='Preview assembly failed: '+str(error)
        with self.lock:self.save()
        self.owner.close()
        print(json.dumps({'event':'recordings_saved','directory':str(self.directory),'pending':self.pending(),
                          'failed':sum(e['state']=='failed' for e in self.entries),
                          'error':self.last_error}),flush=True)
