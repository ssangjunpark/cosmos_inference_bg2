"""One post-pause WAM forecast saved as evidence, with no robot publishing API."""
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from .artifact_protocol import ARTIFACT_API, PUBLIC_FILES, VIDEOS, artifact_path, atomic_json
from .core import BridgeError, require
from .recording import save_inputs
from .rollouts import render_chunk, concatenate
from .rpc import InferenceBusy


EXTRA_VIDEOS = {'imagined_future.mp4': 'rollout_extra.mp4',
                'imagined_head.mp4': 'head_extra.mp4',
                'imagined_wrist_left.mp4': 'wrist_left_extra.mp4',
                'imagined_wrist_right.mp4': 'wrist_right_extra.mp4',
                'rollout_full.mp4': 'rollout_full_extra.mp4'}
NORMAL_VIDEOS = {'imagined_future.mp4': 'rollout.mp4', 'imagined_head.mp4': 'head.mp4',
                 'imagined_wrist_left.mp4': 'wrist_left.mp4', 'imagined_wrist_right.mp4': 'wrist_right.mp4',
                 'rollout_full.mp4': 'rollout.mp4'}


def export_extra_videos(folder, destination, cancelled=lambda: False):
    """Keep normal previews intact and publish separate normal-plus-diagnostic videos."""
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(exist_ok=True)
    for target in EXTRA_VIDEOS.values():
        if (destination/target).exists():
            raise FileExistsError(destination/target)
    manifest=destination/'manifest.json'
    ordinary=[]
    if manifest.is_file():
        import json
        ordinary=json.loads(manifest.read_text())['chunks']
        require(all(item['state'] in ('complete','cancelled') for item in ordinary),
                'Normal recording collection is incomplete; combined diagnostic export deferred')
    if (folder/'decoded_rollout.npy').is_file():
        render_chunk(folder, cancelled)
    outputs = {}
    for source, target in EXTRA_VIDEOS.items():
        if cancelled():
            raise InterruptedError('Pause diagnostic export cancelled')
        path = destination/target
        normal=destination/NORMAL_VIDEOS[source]
        if normal.is_file():
            concatenate([normal,folder/source],path)
        else:
            require(not any(item['state']=='complete' for item in ordinary), 'Original imagined video is missing')
            with (folder/source).open('rb') as incoming, path.open('xb') as outgoing:
                shutil.copyfileobj(incoming,outgoing)
        outputs[target] = str(path)
    return outputs


class PauseDiagnostic:
    def __init__(self, root, observations, status, client, instruction, pause_epoch, *,
                 expected_model=None, cancelled=None, capture_timeout=5.,
                 prediction_timeout=120., recording_wait=30., settle_seconds=.2):
        self.root = Path(root)
        self.directory = self.root/'diagnostics/pause'
        self.observations, self.status, self.client = observations, status, client
        self.artifacts = client.recording_client()
        self.instruction, self.pause_epoch = instruction, pause_epoch
        self.expected_model = expected_model
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.capture_timeout, self.prediction_timeout = capture_timeout, prediction_timeout
        self.recording_wait, self.settle_ns = recording_wait, int(settle_seconds*1e9)
        self.data = {'state': 'pending', 'diagnostic_only': True, 'submitted_to_ffw': False,
                     'executed': False, 'pause_epoch': pause_epoch, 'run_id': self.root.name,
                     'model_id': expected_model, 'instruction': instruction,
                     'settle_seconds': settle_seconds, 'settle_position_tolerance': .01}

    def cancel(self):
        self.cancelled.set()
        self.client.cancel()
        self.artifacts.cancel()

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError('Pause diagnostic cancelled; saved evidence retained')

    def update(self, **fields):
        self.data.update(fields)
        atomic_json(self.directory/'diagnostic.json', self.data)

    def paused(self):
        self.check_cancelled()
        now = time.monotonic_ns()
        cached = self.status.get(now)
        require(cached is not None, 'No fresh FFW status for pause diagnostic')
        current, received = cached
        require(current['mode'] == 'execute' and current['phase'] == 'paused' and
                current['epoch'] == self.pause_epoch and current.get('controllers_ready') is True,
                'FFW is no longer in the confirmed paused session')
        return current, received, now

    def stopped_observation(self):
        current, _, _ = self.paused()
        cutoff = current['robot_now_ns']
        self.update(state='capturing', not_before_source_ns=cutoff)
        deadline = time.monotonic()+self.capture_timeout
        anchor, stable_since, last_source = None, None, 0
        last_error = 'Waiting for fresh post-pause cameras and joints'
        while time.monotonic() < deadline:
            current, received, now = self.paused()
            try:
                observation, raw = self.observations.snapshot(self.instruction,
                    current['robot_now_ns']+now-received, receipt_now_ns=now,
                    not_before_ns=cutoff, include_raw=True)
            except (BridgeError, OSError) as error:
                anchor, stable_since = None, None
                last_error = str(error)
                self.cancelled.wait(.01)
                continue
            source = int(observation['source_time_ns'])
            require(source >= cutoff and all(frame[0] >= cutoff for frame in raw.values()),
                    'Diagnostic observation predates the confirmed pause')
            if source > last_source:
                positions = observation['joint_position']
                if anchor is None or np.max(np.abs(positions-anchor)) > .01:
                    anchor, stable_since = positions.copy(), source
                last_source = source
                if source-stable_since >= self.settle_ns:
                    self.update(source_time_ns=source, settled_since_source_ns=stable_since)
                    return observation, raw
            self.cancelled.wait(.01)
        raise TimeoutError('No settled post-pause observation: '+last_error)

    def prediction(self, observation, request_id):
        deadline = time.monotonic()+self.prediction_timeout
        while True:
            self.paused()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Timed out waiting for post-pause diagnostic inference')
            self.client.timeout = min(120., remaining)
            try:
                return self.client.predict(observation, self.pause_epoch, request_id)
            except InferenceBusy:
                # Ctrl+C cancels the gateway's HTTP wait, not the GPU computation.
                # A 503 explicitly means no new prediction ran, so retry is safe.
                self.update(state='waiting_for_model')
                self.cancelled.wait(.1)

    def collect(self, result):
        reference = result.get('artifacts', {})
        path = artifact_path(self.root.name, self.pause_epoch, result['request_id'])
        require(reference.get('status_path') == path, 'Missing or mismatched diagnostic artifact reference')
        require(reference.get('state') != 'failed', reference.get('error', 'Model recording failed'))
        self.update(state='collecting', artifact_status_path=path)
        deadline = time.monotonic()+self.recording_wait
        while True:
            self.check_cancelled()
            status = self.artifacts.recording_status(path)
            require((status.get('run_id'), status.get('epoch'), status.get('request_id')) ==
                    (self.root.name, self.pause_epoch, result['request_id']),
                    'Diagnostic artifact identity mismatch')
            require(status.get('state') != 'failed', status.get('error', 'Diagnostic recording failed'))
            if status.get('state') == 'complete':
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic prediction saved, but artifact collection did not finish')
            self.cancelled.wait(.1)
        files = status['files']
        require(all(name in PUBLIC_FILES for name in files), 'Unknown diagnostic recording file')
        require('decoded_rollout.npy' in files or all(name in files for name in VIDEOS),
                'Diagnostic recording has no complete imagined rollout')
        folder = self.directory/'artifacts'
        folder.mkdir()
        wanted = {'decoded_rollout.npy', 'rollout.json', 'model_output.json'}
        if 'decoded_rollout.npy' not in files:
            wanted.update(VIDEOS)
        for name, expected in files.items():
            if name in wanted:
                self.check_cancelled()
                self.artifacts.download(path+'/'+name, folder/name, expected)
        return folder

    def run(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.update()
        try:
            self.check_cancelled()
            observation, raw = self.stopped_observation()
            save_inputs(self.directory/'inputs', observation, raw)
            health = self.client.health()
            require(health.get('mode') == 'wam' and health.get('artifacts_api') == ARTIFACT_API,
                    'Pause diagnostic requires the WAM recording API')
            if self.expected_model is not None:
                require(health.get('model_id') == self.expected_model, 'Diagnostic model identity changed')
            request_id = uuid.uuid4().hex
            self.update(state='predicting', request_id=request_id, model_id=health.get('model_id'),
                        checkpoint=health.get('checkpoint'), server_url=self.client.url)
            result = self.prediction(observation, request_id)
            require(result.get('mode') == 'wam' and result.get('epoch') == self.pause_epoch and
                    result.get('request_id') == request_id and
                    result.get('source_time_ns') == int(observation['source_time_ns']),
                    'Diagnostic prediction identity/timestamp mismatch')
            positions = np.asarray(result.get('positions'), dtype=np.float64)
            require(positions.shape == (16, 8) and np.isfinite(positions).all(), 'Invalid diagnostic action chunk')
            atomic_json(self.directory/'prediction.json', {**result, 'diagnostic_only': True,
                                                         'submitted_to_ffw': False, 'executed': False})
            with (self.directory/'joint_positions.npy').open('xb') as stream:
                np.save(stream, positions, allow_pickle=False)
            self.update(state='prediction_saved')
            folder = self.collect(result)
            self.update(state='forecast_saved',artifact_directory=str(folder.relative_to(self.root)))
        except Exception as error:
            state = 'cancelled' if self.cancelled.is_set() or isinstance(error, InterruptedError) else 'failed'
            self.update(state=state, error=str(error))
        finally:
            self.client.cancel()
            self.artifacts.cancel()
        return dict(self.data)

    def finalize(self):
        """Run after ordinary videos finish, so *_extra contains original + extra."""
        if self.data['state']!='forecast_saved':
            return dict(self.data)
        try:
            self.check_cancelled()
            self.update(state='rendering')
            videos=export_extra_videos(self.directory/'artifacts',self.root/'imagined',self.cancelled.is_set)
            self.update(state='complete',video_layout='original_plus_pause_forecast',
                videos={name:str(Path(path).relative_to(self.root)) for name,path in videos.items()})
        except Exception as error:
            self.update(state='cancelled' if self.cancelled.is_set() or isinstance(error,InterruptedError) else 'failed',
                        error=str(error))
        return dict(self.data)
