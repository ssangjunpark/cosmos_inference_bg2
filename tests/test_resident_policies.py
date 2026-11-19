"""CPU-only checks for resident-policy supervision and independent camera capture."""
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
import signal
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]

from PIL import Image

import gateway
import serve_yellowstick_pair as pair
from bg2_bridge.core import BridgeError
from bg2_bridge.gateway import GatewayTransport
import bg2_bridge.recording as recording
from inference_server import PolicyService
from test_gateway import profile
from test_network_inference import FakePolicy, observation


def plans():
    return [pair.ModelPlan('v1', 18420, 18008, 'edge-v1', ROOT/'outputs/v1', ['docker', 'run', 'v1']),
            pair.ModelPlan('v2', 18340, 18009, 'edge-v2', ROOT/'outputs/v2', ['docker', 'run', 'v2'])]


def process():
    value = mock.Mock(returncode=None)
    value.poll.return_value = None
    return value


class DualLauncherTests(unittest.TestCase):
    def test_models_load_serially_and_shutdown_cleans_both_in_reverse_order(self):
        stopped = threading.Event()
        children = [process(), process()]
        sequence = []

        def start(command, **kwargs):
            sequence.append(('start', command[-1]))
            self.assertIn('--cidfile', command)
            self.assertTrue(kwargs['start_new_session'])
            return children[len([event for event in sequence if event[0] == 'start'])-1]

        def ready(plan, *_):
            sequence.append(('ready', plan.model_id))
            if plan.model_id == 'v2':
                stopped.set()
            return True

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(pair, 'ROOT', Path(temporary)), \
                mock.patch.object(pair.subprocess, 'Popen', side_effect=start), \
                mock.patch.object(pair, 'wait_ready', side_effect=ready), \
                mock.patch.object(pair, 'stop_owned') as cleanup, redirect_stdout(StringIO()) as output:
            pair.supervise(plans(), stopped, 10)
        self.assertEqual(sequence, [('start', 'v1'), ('ready', 'v1'), ('start', 'v2'), ('ready', 'v2')])
        self.assertEqual([call.args[0] for call in cleanup.call_args_list], children[::-1])
        self.assertIn('dual_ready', output.getvalue())

    def test_second_model_failure_cleans_first_and_never_reports_dual_ready(self):
        children = [process(), process()]
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(pair, 'ROOT', Path(temporary)), \
                mock.patch.object(pair.subprocess, 'Popen', side_effect=children), \
                mock.patch.object(pair, 'wait_ready', side_effect=[True, RuntimeError('out of memory')]), \
                mock.patch.object(pair, 'stop_owned') as cleanup, redirect_stdout(StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                pair.supervise(plans(), threading.Event(), 10)
        self.assertEqual([call.args[0] for call in cleanup.call_args_list], children[::-1])
        self.assertNotIn('dual_ready', output.getvalue())

    def test_interrupt_during_first_warmup_does_not_start_second_model(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(pair, 'ROOT', Path(temporary)), \
                mock.patch.object(pair.subprocess, 'Popen', return_value=process()) as start, \
                mock.patch.object(pair, 'wait_ready', return_value=False), \
                mock.patch.object(pair, 'stop_owned') as cleanup, redirect_stdout(StringIO()):
            pair.supervise(plans(), threading.Event(), 10)
        start.assert_called_once()
        cleanup.assert_called_once()

    def test_wrong_model_on_endpoint_is_not_accepted_as_ready(self):
        status = {'ok': True, 'warmed': True, 'model_id': 'v2', 'checkpoint': {'step': 18340}}
        with mock.patch.object(pair, 'health', return_value=status):
            with self.assertRaisesRegex(RuntimeError, 'different checkpoint'):
                pair.wait_ready(plans()[0], process(), threading.Event(), 10)

    def test_warmup_timeout_does_not_wait_forever(self):
        with mock.patch.object(pair, 'health', side_effect=URLError('not ready')), \
                mock.patch.object(pair.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                pair.wait_ready(plans()[0], process(), threading.Event(), 1)

    def test_child_exit_is_detected_without_contacting_a_stale_endpoint(self):
        child = process()
        child.poll.return_value = child.returncode = 2
        with mock.patch.object(pair, 'health') as health:
            with self.assertRaisesRegex(RuntimeError, 'exited before'):
                pair.wait_ready(plans()[0], child, threading.Event(), 10)
        health.assert_not_called()

    def test_cleanup_uses_created_container_id_instead_of_container_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            cidfile = Path(temporary)/'v1.cid'
            cidfile.write_text('a'*64)
            child = process()
            with mock.patch.object(pair.subprocess, 'run', return_value=NS(returncode=0, stderr='')) as stop:
                pair.stop_owned(child, cidfile)
            self.assertEqual(stop.call_args.args[0], ['docker', 'stop', '--timeout', '20', 'a'*64])
            child.terminate.assert_not_called()
            child.wait.assert_called_once()

    def test_cleanup_handles_container_id_created_during_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cidfile = Path(temporary)/'v1.cid'
            child = process()
            child.wait.side_effect = lambda **_: cidfile.write_text('b'*64)
            with mock.patch.object(pair.subprocess, 'run', return_value=NS(returncode=0, stderr='')) as stop:
                pair.stop_owned(child, cidfile)
            child.terminate.assert_called_once()
            self.assertEqual(stop.call_args.args[0][-1], 'b'*64)

    def test_existing_named_container_is_not_stopped_or_reused(self):
        with mock.patch.object(pair.subprocess, 'run', return_value=NS(stdout='edge-v1\n')) as run:
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                pair.check_environment(plans(), '0', 26)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:2], ['docker', 'ps'])

    def test_insufficient_memory_is_rejected_before_any_start(self):
        with mock.patch.object(pair.subprocess, 'run', side_effect=[NS(stdout=''), NS(stdout=str(20*1024))]), \
                mock.patch.object(pair.socket, 'socket'):
            with self.assertRaisesRegex(RuntimeError, 'requires 26 GiB'):
                pair.check_environment(plans(), '0', 26)

    def test_configs_keep_weights_outputs_and_endpoints_distinct(self):
        config = json.loads((ROOT/'deploy/5090/yellowstick_pair.json').read_text())
        with mock.patch.object(pair.cosmos3, 'server_command', return_value=['docker', 'run']) as build:
            result = pair.model_plans(config)
        self.assertEqual([(p.model_id, p.step, p.port) for p in result], [('v1', 18420, 18008), ('v2', 18340, 18009)])
        self.assertNotEqual(result[0].name, result[1].name)
        self.assertNotEqual(result[0].output, result[1].output)
        self.assertNotEqual(build.call_args_list[0].args[0].model_dir, build.call_args_list[1].args[0].model_dir)
        config['models'][1]['port'] = config['models'][0]['port']
        with mock.patch.object(pair.cosmos3, 'server_command', return_value=['docker', 'run']):
            with self.assertRaisesRegex(ValueError, 'distinct port'):
                pair.model_plans(config)

    def test_dry_run_never_checks_gpu_docker_ports_or_health(self):
        with mock.patch.object(pair, 'model_plans', return_value=plans()), \
                mock.patch.object(pair, 'check_environment') as check, \
                mock.patch.object(pair, 'supervise') as start, \
                mock.patch.object(pair.signal, 'signal'), redirect_stdout(StringIO()) as output:
            self.assertEqual(pair.main(['--dry-run']), 0)
        check.assert_not_called()
        start.assert_not_called()
        self.assertIn('dual_plan_valid', output.getvalue())


class CameraAndIdentityTests(unittest.TestCase):
    def test_health_identifies_loaded_checkpoint(self):
        class TaggedPolicy(FakePolicy):
            runtime_metadata = {'checkpoint': {'dataset_variant': 'v2', 'step': 18340}}
        service = PolicyService(TaggedPolicy, {}, observation(), 1)
        try:
            self.assertEqual(service.health()['model_id'], 'v2')
            self.assertEqual(service.health()['checkpoint']['step'], 18340)
        finally:
            service.close()

    def test_wrong_expected_model_opens_no_robot_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary)/'profile.json'
            config.write_text(json.dumps(profile()))
            with mock.patch.object(gateway, 'InferenceClient') as client, \
                    mock.patch.object(gateway, 'GatewayTransport') as transport:
                client.return_value.health.return_value = {'model_id': 'v1'}
                with self.assertRaisesRegex(BridgeError, 'Expected model v2'):
                    gateway.main(['execute', '--debug', '--profile', str(config), '--instruction', 'task',
                                  '--expected-model', 'v2', '--output', str(Path(temporary)/'run')])
                transport.assert_not_called()

    def test_recorder_subscribes_only_to_cameras_and_finalizes_on_interrupt(self):
        subscribers, written, closed, handlers = [], {}, [], {}

        class Video:
            def __init__(self, _, key):
                self.key, self.count = key, 0
                written[key] = []

            def write(self, source, receipt, payload):
                written[self.key].append((source, receipt, payload))
                self.count += 1

            def close(self):
                closed.append(self.key)

        class Subscriber:
            def __init__(self, **kwargs):
                subscribers.append(kwargs)

            def close(self):
                pass

        p = profile()
        sdk = NS(ROS2Subscriber=Subscriber, get_registry=mock.Mock(),
                 ROS2Publisher=mock.Mock(side_effect=AssertionError('Recorder must have no publishers')))
        expected = {}

        def connect(profile, observations, status, **kwargs):
            self.assertTrue(kwargs['inspect'])
            self.assertTrue(kwargs['cameras_only'])
            transport = GatewayTransport(profile, observations, status, sdk=sdk, **kwargs)
            for key, camera in p['cameras'].items():
                image = BytesIO()
                Image.new('RGB', tuple(camera['raw_wh']), (25, 50, 75)).save(image, format='JPEG')
                payload = image.getvalue()
                expected[key] = payload
                callback = next(s['callback'] for s in subscribers if s['topic'] == camera['topic'])
                callback(NS(header=NS(stamp=NS(sec=1, nanosec=23)), data=payload))
            self.assertIsNone(transport.chunk_publisher)
            self.assertIsNone(transport.control_publisher)
            handlers[signal.SIGINT]()
            return transport

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary)/'profile.json'
            config.write_text(json.dumps(p))
            output = Path(temporary)/'continuous'
            with mock.patch.object(recording, 'CameraVideo', Video), \
                    mock.patch.object(gateway, 'InferenceClient', side_effect=AssertionError('No model requests')), \
                    mock.patch.object(gateway, 'GatewayTransport', side_effect=connect), \
                    mock.patch.object(gateway.signal, 'signal', side_effect=lambda sig, handler: handlers.update({sig: handler})):
                gateway.main(['record-cameras', '--profile', str(config), '--output', str(output)])
            metadata = json.loads((output/'run.json').read_text())
            summary = json.loads((output/'recording_summary.json').read_text())
            self.assertEqual(metadata['mode'], 'record-cameras')
            self.assertEqual(metadata['episode_state'], 'ended')
            self.assertFalse(metadata['physical_publishing'])
            self.assertEqual(summary['frames'], dict.fromkeys(p['cameras'], 1))
            self.assertFalse((output/'predictions').exists())
        self.assertEqual(len(subscribers), 3)
        self.assertEqual({s['msg_type'] for s in subscribers}, {'sensor_msgs/msg/CompressedImage'})
        self.assertCountEqual(closed, p['cameras'])
        for key in p['cameras']:
            self.assertEqual(written[key][0][0], 1_000_000_023)
            self.assertEqual(written[key][0][2], expected[key])
        sdk.ROS2Publisher.assert_not_called()


if __name__ == '__main__':
    unittest.main()
