"""Durable WAM snapshots and recording manifests; MP4s are built later on 5090."""
import hashlib
import json
from pathlib import Path
import queue
import threading

import numpy as np

from bg2_bridge.artifact_protocol import ARTIFACT_API, PUBLIC_FILES, artifact_path, atomic_json, run_name
from bg2_bridge.rpc import identifier


class ModelRecordings:
    def __init__(self, root, renderer=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer or self.render
        self.jobs = queue.Queue()
        self.stop_event = threading.Event()
        self.last_error = None
        self.thread = threading.Thread(target=self.work, name='imagined-recording-manifests', daemon=True)
        for status in sorted(self.root.glob('runs/*/requests/*/*/recording.json')):
            if json.loads(status.read_text()).get('state') in ('pending', 'rendering'):
                self.jobs.put(status.parent)
        self.thread.start()

    def folder(self, run_id, epoch, request_id):
        return self.root / 'runs' / run_name(run_id) / 'requests' / identifier(epoch) / identifier(request_id)

    def submit(self, observation, result, response, sampling, run_id):
        folder = self.folder(run_id, response['epoch'], response['request_id'])
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'inputs').mkdir()
        # Only raw snapshots and a small journal are synchronous. No PNG compression,
        # MP4 rendering, ffprobe, downloads or file hashing delay the action response.
        with (folder / 'inputs/observation.npz').open('xb') as stream:
            np.savez(stream, **observation)
        arrays = {'normalized_actions.npy': result['normalized'], 'raw_joint_positions.npy': result['positions'],
                  'joint_positions.npy': np.asarray(response['positions']), 'decoded_rollout.npy': result['imagined_video']}
        for index, latent in enumerate(result.get('vision', [])):
            arrays[f'vision_{index:03d}_latent.npy'] = latent
        for name, value in arrays.items():
            with (folder / name).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
        atomic_json(folder / 'prediction.json', response)
        atomic_json(folder / 'model_output.json', {'mode': 'wam', 'imagined_future_rollout': True,
            'prompt': result.get('prompt'), 'sampling': sampling, 'runtime_metadata': result.get('runtime_metadata'),
            'note': 'Raw actions preserved; returned commands floor negative right-gripper values at zero.'})
        reference = {'api': ARTIFACT_API, 'run_id': run_id, 'epoch': response['epoch'],
                     'request_id': response['request_id'], 'directory': folder.relative_to(self.root).as_posix(),
                     'status_path': artifact_path(run_id, response['epoch'], response['request_id']), 'state': 'pending'}
        atomic_json(folder / 'recording.json', {'ok': True, **reference, 'files': {}})
        self.jobs.put(folder)
        return reference

    @staticmethod
    def render(folder):
        with np.load(folder/'inputs/observation.npz',allow_pickle=False) as observation:
            source=int(observation['source_time_ns'])
        offsets=[i*50_000_000 for i in range(17)]
        atomic_json(folder/'rollout.json',{'fps':20,'frame_count':17,'source_time_ns':source,
            'frame_offsets_ns':offsets,'forecast_time_ns':[source+offset for offset in offsets],
            'context_frame':0,'future_frames':list(range(1,17)),
            'note':'Frame 0 is reconstructed context. Forecast times are relative to the observation, not robot execution.'})

    def work(self):
        while not self.stop_event.is_set():
            try:
                folder = self.jobs.get(timeout=.2)
            except queue.Empty:
                continue
            status_path = folder / 'recording.json'
            try:
                data = json.loads(status_path.read_text())
                atomic_json(status_path, {**data, 'state': 'rendering'})
                self.renderer(folder)
                files = {}
                for name in PUBLIC_FILES:
                    path = folder / name
                    if path.exists():
                        files[name] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                atomic_json(status_path, {**data, 'state': 'complete', 'files': files})
            except Exception as error:
                self.last_error = str(error)
                try:
                    atomic_json(status_path, {**data, 'state': 'failed', 'error': str(error)})
                except Exception:
                    pass
            finally:
                self.jobs.task_done()

    def status(self, run_id, epoch, request_id):
        folder=self.lookup_folder(run_id,epoch,request_id)
        if (folder/'recording.json').exists():
            return json.loads((folder/'recording.json').read_text())
        # Existing completed recordings can be collected without moving or rewriting them.
        metadata=folder/'model_output.json'
        data=json.loads((metadata if metadata.exists() else folder/'prediction.json').read_text())
        if data.get('mode')!='wam':raise FileNotFoundError('No imagined rollout for this policy recording')
        from bg2_bridge.artifact_protocol import VIDEOS
        videos_complete=metadata.exists() and all((folder/name).is_file() for name in VIDEOS)
        raw=(folder/'decoded_rollout.npy').is_file()
        if raw and not (folder/'rollout.json').exists():self.render(folder)
        complete=videos_complete or raw
        files={name:{'bytes':(folder/name).stat().st_size,
                     'sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest()}
               for name in PUBLIC_FILES if (folder/name).is_file()
               and not (videos_complete and name=='decoded_rollout.npy')
               and not (not videos_complete and name in VIDEOS)}
        return {'ok':True,'api':ARTIFACT_API,'run_id':run_id,'epoch':epoch,'request_id':request_id,
                'state':'complete' if complete else 'failed','files':files,
                'error':None if complete else 'Legacy video export was incomplete; raw snapshot is retained.'}

    def lookup_folder(self,run_id,epoch,request_id):
        folder=self.folder(run_id,epoch,request_id)
        return folder if folder.exists() else self.root/'requests'/identifier(epoch)/identifier(request_id)

    def find(self,run_id,request_id):
        run_name(run_id);identifier(request_id)
        matches=list(self.root.glob('requests/*/'+request_id+'/prediction.json'))
        matches+=list((self.root/'runs'/run_id).glob('requests/*/'+request_id+'/prediction.json'))
        if len(matches)!=1:raise FileNotFoundError('No unique saved prediction')
        data=json.loads(matches[0].read_text())
        return {'ok':True,'run_id':run_id,'epoch':data['epoch'],'request_id':request_id,
                'source_time_ns':data['source_time_ns'],'status_path':artifact_path(run_id,data['epoch'],request_id)}

    def file(self, run_id, epoch, request_id, name):
        if name not in PUBLIC_FILES:
            raise ValueError('Unknown recording file')
        state = self.status(run_id, epoch, request_id)
        if state['state'] != 'complete' or name not in state['files']:
            raise FileNotFoundError('Recording is not ready')
        return self.lookup_folder(run_id, epoch, request_id) / name

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)
