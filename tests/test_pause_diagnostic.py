"""Post-pause forecasts use fresh state and never publish a diagnostic chunk."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]

import gateway as cli
from bg2_bridge.artifact_protocol import ARTIFACT_API, artifact_path
from bg2_bridge.core import BridgeError
from bg2_bridge.pause_diagnostic import PauseDiagnostic, EXTRA_VIDEOS, export_extra_videos
from bg2_bridge.rpc import InferenceBusy, InferenceClient
from test_gateway import profile, image, status
from test_network_inference import observation


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/'test-run'
        (self.root/'imagined').mkdir(parents=True)
        (self.root/'imagined/head.mp4').write_bytes(b'normal forecast')
        (self.root/'imagined/manifest.json').write_text('{"chunks":["normal"]}')
        self.epoch = uuid.uuid4().hex
        self.cutoff = 10_000_000_000
        self.obs = observation(self.cutoff+10_000_000)
        self.obs['joint_position'][:] = np.arange(8)/10
        self.sensors = mock.Mock()
        self.sensors.snapshot.side_effect = self.snapshot
        self.cache = mock.Mock()
        self.cache.get.return_value = ({'mode': 'execute', 'phase': 'paused', 'epoch': self.epoch,
            'controllers_ready': True, 'robot_now_ns': self.cutoff}, time.monotonic_ns())
        self.client = mock.Mock()
        self.client.url = 'http://127.0.0.1:18009'
        self.client.health.return_value = {'mode': 'wam', 'artifacts_api': ARTIFACT_API,
            'model_id': 'v2', 'checkpoint': {'step': 18340}}
        self.client.predict.side_effect = self.reply
        self.artifacts = self.client.recording_client.return_value
        self.artifacts.recording_status.side_effect = self.recording_status
        self.artifacts.download.side_effect = lambda path, destination, expected: destination.write_bytes(b'raw')
        self.request = None

    def snapshot(self, *args, **kwargs):
        self.assertEqual(kwargs['not_before_ns'], self.cutoff)
        self.assertTrue(kwargs['include_raw'])
        source = int(self.obs['source_time_ns'])
        return self.obs, {key: (source, time.monotonic_ns(), image(key)) for key in ('head', 'wrist_left', 'wrist')}

    def reply(self, observed, epoch, request_id):
        self.request = request_id
        self.assertEqual(epoch, self.epoch)
        return {'mode': 'wam', 'epoch': epoch, 'request_id': request_id,
            'source_time_ns': int(observed['source_time_ns']), 'positions': [[.25]*8]*16,
            'artifacts': {'state': 'pending', 'status_path': artifact_path(self.root.name, epoch, request_id)}}

    def recording_status(self, path):
        return {'state': 'complete', 'run_id': self.root.name, 'epoch': self.epoch,
            'request_id': self.request, 'files': {'decoded_rollout.npy': {'bytes': 3, 'sha256': 'fake'}}}

    def diagnostic(self, **kwargs):
        return PauseDiagnostic(self.root, self.sensors, self.cache, self.client,
            'Move the yellow stick into the basket.', self.epoch, expected_model='v2',
            settle_seconds=kwargs.pop('settle_seconds', 0), **kwargs)

    def run_diagnostic(self, diagnostic=None):
        def export(folder, destination, cancelled):
            outputs = {}
            for target in EXTRA_VIDEOS.values():
                path = destination/target
                path.write_bytes(b'extra forecast')
                outputs[target] = str(path)
            return outputs
        with mock.patch('bg2_bridge.pause_diagnostic.export_extra_videos', side_effect=export):
            diagnostic=diagnostic or self.diagnostic()
            result=diagnostic.run()
            return diagnostic.finalize() if result['state']=='forecast_saved' else result

    def test_one_new_forecast_saves_actions_and_extra_previews_without_touching_normal_rollout(self):
        result = self.run_diagnostic()
        self.assertEqual(result['state'], 'complete')
        self.assertFalse(result['submitted_to_ffw'])
        self.assertFalse(result['executed'])
        self.client.predict.assert_called_once()
        selected = self.client.predict.call_args.args[0]
        np.testing.assert_array_equal(selected['joint_position'], np.arange(8, dtype=np.float32)/10)
        self.assertEqual(int(selected['source_time_ns']), self.cutoff+10_000_000)
        saved = json.loads((self.root/'diagnostics/pause/prediction.json').read_text())
        self.assertTrue(saved['diagnostic_only'])
        self.assertFalse(saved['submitted_to_ffw'])
        np.testing.assert_array_equal(np.load(self.root/'diagnostics/pause/joint_positions.npy'), [[.25]*8]*16)
        self.assertTrue((self.root/'diagnostics/pause/inputs/input_composite.png').is_file())
        self.assertEqual((self.root/'imagined/head.mp4').read_bytes(), b'normal forecast')
        self.assertEqual(json.loads((self.root/'imagined/manifest.json').read_text()), {'chunks': ['normal']})
        self.assertFalse((self.root/'predictions').exists())
        self.assertTrue((self.root/'imagined/head_extra.mp4').is_file())

    def test_busy_refusal_retries_same_request_but_runs_only_one_accepted_prediction(self):
        calls = 0
        def busy_then_ready(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InferenceBusy('Model is busy')
            return self.reply(*args)
        self.client.predict.side_effect = busy_then_ready
        self.assertEqual(self.run_diagnostic()['state'], 'complete')
        self.assertEqual(self.client.predict.call_count, 2)
        first, second = self.client.predict.call_args_list
        self.assertEqual(first.args[1:], second.args[1:])
        self.assertIs(first.args[0], second.args[0])

    def test_uncertain_inference_failure_is_not_retried(self):
        self.client.predict.side_effect = ConnectionResetError('reply lost')
        result = self.run_diagnostic()
        self.assertEqual(result['state'], 'failed')
        self.client.predict.assert_called_once()
        self.artifacts.recording_status.assert_not_called()
        self.assertTrue((self.root/'diagnostics/pause/inputs/observation.npz').exists())

    def test_missing_or_changed_paused_session_cannot_request_a_prediction(self):
        self.cache.get.return_value = None
        result = self.run_diagnostic()
        self.assertEqual(result['state'], 'failed')
        self.client.predict.assert_not_called()
        self.client.health.assert_not_called()

    def test_pre_pause_frames_are_never_reused(self):
        self.obs['source_time_ns'] = np.asarray(self.cutoff-1, dtype=np.int64)
        result = self.run_diagnostic()
        self.assertEqual(result['state'], 'failed')
        self.assertIn('predates', result['error'])
        self.client.predict.assert_not_called()

    def test_moving_feedback_restarts_the_stability_window(self):
        samples = iter([(100_000_000, 0.), (250_000_000, .1), (350_000_000, .1), (500_000_000, .1)])
        def snapshots(*args, **kwargs):
            offset, position = next(samples)
            self.obs['source_time_ns'] = np.asarray(self.cutoff+offset)
            self.obs['joint_position'][:] = position
            return self.snapshot(*args, **kwargs)
        self.sensors.snapshot.side_effect = snapshots
        result = self.run_diagnostic(self.diagnostic(settle_seconds=.2))
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['source_time_ns'], self.cutoff+500_000_000)
        self.assertEqual(result['settled_since_source_ns'], self.cutoff+250_000_000)

    def test_cancel_before_capture_sends_no_request(self):
        cancelled = threading.Event()
        cancelled.set()
        result = self.run_diagnostic(self.diagnostic(cancelled=cancelled))
        self.assertEqual(result['state'], 'cancelled')
        self.client.predict.assert_not_called()

    def test_cancel_after_prediction_retains_actions_without_exporting(self):
        cancelled = threading.Event()
        def response(*args):
            result = self.reply(*args)
            cancelled.set()
            return result
        self.client.predict.side_effect = response
        result = self.run_diagnostic(self.diagnostic(cancelled=cancelled))
        self.assertEqual(result['state'], 'cancelled')
        self.assertTrue((self.root/'diagnostics/pause/joint_positions.npy').is_file())
        self.assertFalse((self.root/'imagined/head_extra.mp4').exists())

    def test_artifacts_from_another_request_are_rejected(self):
        self.artifacts.recording_status.return_value = {}
        self.artifacts.recording_status.side_effect = lambda _: {'state': 'complete', 'request_id': uuid.uuid4().hex}
        result = self.run_diagnostic()
        self.assertEqual(result['state'], 'failed')
        self.assertIn('identity mismatch', result['error'])
        self.artifacts.download.assert_not_called()


class DiagnosticCliTests(unittest.TestCase):
    def run_cli(self, *, enabled=True, signum=signal.SIGINT, confirmed=True):
        handlers, events = {}, []
        p = profile()
        p.update(execution_enabled=True, status='operator_enabled_for_execution')
        paused_epoch = uuid.uuid4().hex
        transport = mock.Mock()
        transport.diagnostics.return_value = {}
        recorder = mock.Mock(metadata={})
        cache = None

        def connect(profile, observations, status_cache, **kwargs):
            nonlocal cache
            cache = status_cache
            cache.update(status(uuid.uuid4().hex, mode='execute', phase='holding'), time.monotonic_ns())
            return transport

        def control(epoch, command):
            events.append(command)
            cache.update({**status(paused_epoch, mode='execute', phase='paused', ready_since=0),
                          'controllers_ready': True}, time.monotonic_ns())
        transport.control.side_effect = control

        class Engine:
            def __init__(self, *args, **kwargs):
                pass
            def close(self):
                pass
            def step(self, *_):
                handlers[signum](signum)
                return {'event': 'interrupted'}

        client = mock.Mock()
        client.health.return_value = {'records_outputs': True, 'mode': 'wam',
            'artifacts_api': ARTIFACT_API, 'model_id': 'v2'}
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary)/'profile.json'
            config.write_text(json.dumps(p))
            output = Path(temporary)/'run'
            with mock.patch.object(cli, 'GatewayTransport', side_effect=connect), \
                    mock.patch.object(cli, 'Gateway', Engine), \
                    mock.patch.object(cli, 'RunRecorder', return_value=recorder), \
                    mock.patch.object(cli, 'InferenceClient', return_value=client) as factory, \
                    mock.patch.object(cli.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
                    mock.patch('bg2_bridge.pause_diagnostic.PauseDiagnostic') as diagnostic, \
                    redirect_stdout(StringIO()):
                def extra():
                    events.append('diagnostic')
                    return {'state': 'forecast_saved', 'executed': False}
                diagnostic.return_value.run.side_effect = extra
                diagnostic.return_value.finalize.return_value={'state':'complete','executed':False}
                args = ['execute', '--profile', str(config), '--instruction', 'task',
                        '--expected-model', 'v2', '--output', str(output)]
                if enabled:
                    args.append('--pause-diagnostic')
                if confirmed:
                    cli.main(args)
                else:
                    with mock.patch.object(cli, 'finish_control', return_value={'event': 'control_not_confirmed'}):
                        cli.main(args)
                calls, diagnostic_calls = factory.call_count, diagnostic.call_count
                if diagnostic_calls:
                    self.assertEqual(diagnostic.call_args.args[5], paused_epoch)
                transport.send_chunk.assert_not_called()
                transport.close.assert_called_once()
                recorder.close.assert_called_once()
        return events, calls, diagnostic_calls

    def test_ctrl_c_runs_diagnostic_after_pause_and_never_sends_a_chunk(self):
        self.assertEqual(self.run_cli(), (['pause', 'diagnostic'], 2, 1))

    def test_feature_is_off_by_default(self):
        self.assertEqual(self.run_cli(enabled=False), (['pause'], 1, 0))

    def test_sigterm_does_not_trigger_extra_inference(self):
        self.assertEqual(self.run_cli(signum=signal.SIGTERM), (['pause'], 1, 0))

    def test_unconfirmed_pause_does_not_trigger_extra_inference(self):
        self.assertEqual(self.run_cli(confirmed=False), ([], 1, 0))

    def test_debug_and_camera_recording_reject_the_execute_only_option(self):
        for mode, options in [('execute', ['--debug']), ('record-cameras', []), ('preview', [])]:
            with self.subTest(mode=mode), self.assertRaises(BridgeError):
                cli.main([mode, *options, '--pause-diagnostic', '--output', '/tmp/unused-pause-diagnostic'])


class BusyReplyTests(unittest.TestCase):
    def test_busy_http_reply_has_a_distinct_exception_for_safe_retry(self):
        response = mock.Mock(status=503)
        response.read.return_value = b'{"ok":false,"error":"Model is busy"}'
        connection = mock.Mock()
        connection.getresponse.return_value = response
        with mock.patch('bg2_bridge.rpc.http.client.HTTPConnection', return_value=connection):
            with self.assertRaises(InferenceBusy):
                InferenceClient('http://127.0.0.1:18009')._call('POST', '/predict')


@unittest.skipUnless(importlib.util.find_spec('av'), 'PyAV required for media validation')
class DiagnosticMediaTests(unittest.TestCase):
    def test_extra_mp4s_have_all_future_frames_and_leave_normal_videos_unchanged(self):
        import av
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, imagined = root/'artifacts', root/'imagined'
            folder.mkdir()
            imagined.mkdir()
            (imagined/'timeline.json').write_text('{"frame_count":32}')
            from bg2_bridge.rollouts import render_chunk,concatenate
            from bg2_bridge.pause_diagnostic import NORMAL_VIDEOS
            normal=root/'normal';normal.mkdir()
            ordinary=np.full((17,640,512,3),[220,0,0],np.uint8)
            np.save(normal/'decoded_rollout.npy',ordinary,allow_pickle=False)
            render_chunk(normal)
            for source,target in NORMAL_VIDEOS.items():
                if target=='rollout.mp4' and (imagined/target).exists():continue
                concatenate([normal/source,normal/source],imagined/target)
            originals={path.name:path.read_bytes() for path in imagined.glob('*.mp4')}
            frames = np.zeros((17, 640, 512, 3), np.uint8)
            frames[0] = [220, 0, 0]
            frames[1:, :384] = [0, 220, 0]
            frames[1:, 384:, :256] = [0, 0, 220]
            frames[1:, 384:, 256:] = [220, 220, 0]
            np.save(folder/'decoded_rollout.npy', frames, allow_pickle=False)
            output = export_extra_videos(folder, imagined)
            self.assertEqual(set(output), set(EXTRA_VIDEOS.values()))
            for name in EXTRA_VIDEOS.values():
                with av.open(str(imagined/name)) as video:
                    decoded = list(video.decode(video=0))
                    self.assertEqual(len(decoded), 49 if name == 'rollout_full_extra.mp4' else 48)
                    self.assertEqual(float(video.streams.video[0].average_rate), 20.)
            with av.open(str(imagined/'head_extra.mp4')) as video:
                decoded=list(video.decode(video=0))
                self.assertGreater(int(decoded[0].to_ndarray(format='rgb24')[20,20,0]),200)
                self.assertGreater(int(decoded[31].to_ndarray(format='rgb24')[20,20,0]),200)
                pixels = decoded[32].to_ndarray(format='rgb24')
                self.assertEqual(pixels.shape, (384, 512, 3))
                self.assertGreater(int(pixels[20,20,1]), 200)
            self.assertEqual(originals,{name:(imagined/name).read_bytes() for name in originals})
            self.assertEqual(json.loads((imagined/'timeline.json').read_text()), {'frame_count': 32})
            with self.assertRaises(FileExistsError):
                export_extra_videos(folder, imagined)


if __name__ == '__main__':
    unittest.main()
