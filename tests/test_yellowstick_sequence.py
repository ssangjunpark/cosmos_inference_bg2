"""CPU-only checks for Enter-controlled v2/v1 and owned recorder cleanup."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

from bg2_bridge.artifact_protocol import atomic_json
from bg2_bridge.pause_diagnostic import EXTRA_VIDEOS
import run_yellowstick_sequence as sequence


def camera_receipt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder/'run.json', {'episode_state': 'ended', 'pause_diagnostic': {'state': 'complete'}})
    atomic_json(folder/'recording_summary.json',
                {'frames': {'head': 1, 'wrist_left': 1, 'wrist': 1}, 'error': None, 'close_errors': []})
    (folder/'cameras').mkdir(exist_ok=True)
    for key in ('head', 'wrist_left', 'wrist'):
        (folder/'cameras'/(key+'.mkv')).write_bytes(b'finished camera data')
        (folder/'cameras'/(key+'.timestamps.jsonl')).write_text('{"frame":0,"source_time_ns":1}\n')


def gateway_receipt(folder, *, diagnostic='complete'):
    camera_receipt(folder)
    epoch = uuid.uuid4().hex
    (folder/'diagnostics/pause').mkdir(parents=True)
    atomic_json(folder/'diagnostics/pause/diagnostic.json',
                {'state': diagnostic, 'pause_epoch': epoch, 'executed': False, 'submitted_to_ffw': False,
                 'video_layout':'original_plus_pause_forecast'})
    (folder/'events.jsonl').write_text(json.dumps({'event': 'pause_confirmed', 'epoch': epoch})+'\n')
    (folder/'imagined').mkdir()
    atomic_json(folder/'imagined/manifest.json', {'chunks': [], 'last_error': None})
    for name in EXTRA_VIDEOS.values():
        (folder/'imagined'/name).write_bytes(b'finished MP4')


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        patcher = mock.patch.object(sequence, 'OUTPUTS', self.base)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.console = NS(abort=False, interrupts=0, enter=mock.Mock(return_value=True), discard_pending=mock.Mock())
        self.selected = {'v1': sequence.Policy('v1', 18420, 'profiles/v1.json', 'http://127.0.0.1:18008'),
                         'v2': sequence.Policy('v2', 18340, 'profiles/v2.json', 'http://127.0.0.1:18009')}
        self.session = sequence.Sequence(self.selected, self.base/'session', self.console,
                                         instruction='Move the yellow stick into the basket.')

    def run_simulation(self, *, diagnostic='complete', abort_after_v2=False):
        events, commands = [], []
        def start(role, name, folder, command, profile):
            events.append(('start', role))
            commands.append((role, command, profile))
            process = mock.Mock(returncode=0)
            process.poll.return_value = None
            process.wait.return_value = 0
            child = sequence.Child(role, name, folder, process)
            self.session.children.append(child)
            return child

        def pause(child, **kwargs):
            events.append(('pause', child.role))
            child.interrupts += 1
            gateway_receipt(child.folder, diagnostic=diagnostic)
            child.process.poll.return_value = 0
            if abort_after_v2:
                self.console.abort = True

        def finish(child):
            events.append(('finalized', child.role))
            if child.role == 'continuous-cameras':
                camera_receipt(child.folder)
            child.process.poll.return_value = 0

        with mock.patch.object(self.session, 'preflight'), \
                mock.patch.object(self.session, 'start', side_effect=start), \
                mock.patch.object(self.session, 'wait_ready'), \
                mock.patch.object(self.session, 'request_pause', side_effect=pause), \
                mock.patch.object(self.session, 'stop_and_wait', side_effect=finish), redirect_stdout(StringIO()):
            if diagnostic == 'complete':
                self.session.run()
            else:
                with self.assertRaisesRegex(RuntimeError, 'diagnostic did not finish'):
                    self.session.run()
        return events, commands

    def test_v2_finishes_before_v1_and_continuous_cameras_finish_last(self):
        events, commands = self.run_simulation()
        self.assertLess(events.index(('start', 'continuous-cameras')), events.index(('start', 'v2')))
        self.assertLess(events.index(('finalized', 'v2')), events.index(('start', 'v1')))
        self.assertLess(events.index(('finalized', 'v1')), events.index(('finalized', 'continuous-cameras')))
        self.assertEqual([role for role, _, _ in commands], ['continuous-cameras', 'v2', 'v1'])
        for role, command, profile in commands:
            if role in ('v1', 'v2'):
                self.assertIn('--pause-diagnostic', command)
                self.assertEqual(command[command.index('--expected-model')+1], role)
                self.assertEqual(command[command.index('--server-url')+1], self.selected[role].url)
                self.assertEqual(profile, self.selected[role].profile)
        self.assertEqual(self.session.metadata['state'], 'complete')
        self.assertEqual(self.console.enter.call_count,2)
        self.assertFalse(self.session.metadata['policy_servers_managed'])
        self.assertFalse(self.session.metadata['robot_bringup_managed'])

    def test_incomplete_v2_diagnostic_prevents_v1_and_still_finalizes_cameras(self):
        events, _ = self.run_simulation(diagnostic='failed')
        self.assertNotIn(('start', 'v1'), events)
        self.assertIn(('finalized', 'continuous-cameras'), events)
        self.assertEqual(self.session.metadata['state'], 'failed')

    def test_parent_ctrl_c_cancels_remaining_policy_after_current_cleanup(self):
        events, _ = self.run_simulation(abort_after_v2=True)
        self.assertNotIn(('start', 'v1'), events)
        self.assertIn(('finalized', 'v2'), events)
        self.assertIn(('finalized', 'continuous-cameras'), events)
        self.assertEqual(self.session.metadata['state'], 'cancelled')

    def test_missing_normal_video_assembly_does_not_count_as_saved(self):
        folder = self.base/'incomplete'
        gateway_receipt(folder)
        atomic_json(folder/'imagined/manifest.json', {'chunks': [{'state': 'pending'}], 'last_error': None})
        with self.assertRaisesRegex(RuntimeError, 'Normal recordings are incomplete'):
            sequence.validate_gateway_recording(folder)

    def test_proven_unexecuted_cancelled_request_does_not_require_an_imagined_clip(self):
        folder=self.base/'cancelled-request'
        gateway_receipt(folder)
        cancelled={'state':'cancelled','prediction_received':False,'submitted_to_ffw':False,
                   'cancellation_proof':'later_diagnostic_completed'}
        atomic_json(folder/'imagined/manifest.json',{'chunks':[cancelled],'last_error':None})
        sequence.validate_gateway_recording(folder)
        cancelled['prediction_received']=True
        atomic_json(folder/'imagined/manifest.json',{'chunks':[cancelled],'last_error':None})
        with self.assertRaisesRegex(RuntimeError,'Normal recordings are incomplete'):
            sequence.validate_gateway_recording(folder)

    def test_missing_extra_video_prevents_advance(self):
        folder = self.base/'missing-extra'
        gateway_receipt(folder)
        (folder/'imagined/head_extra.mp4').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing diagnostic preview'):
            sequence.validate_gateway_recording(folder)

    def test_truncated_camera_timestamps_prevent_advance(self):
        folder = self.base/'missing-camera'
        gateway_receipt(folder)
        (folder/'cameras/head.timestamps.jsonl').write_text('')
        with self.assertRaisesRegex(RuntimeError, 'timestamp/frame count differs'):
            sequence.validate_gateway_recording(folder)

    def test_signals_use_owned_container_id_even_if_compose_client_exited(self):
        child = sequence.Child('v2', sequence.GATEWAY_NAME, self.base/'run', mock.Mock(), identity='a'*64)
        child.process.poll.return_value = 0
        with mock.patch.object(sequence, 'inspect_container', return_value={'running': True, 'session': self.session.session_id}), \
                mock.patch.object(sequence.subprocess, 'run') as docker:
            self.session.request_pause(child)
        self.assertEqual(docker.call_args.args[0], ['docker', 'kill', '--signal=SIGINT', 'a'*64])
        self.assertEqual(child.interrupts, 1)

    def test_foreign_container_cannot_receive_a_signal(self):
        child = sequence.Child('v2', sequence.GATEWAY_NAME, self.base/'run', mock.Mock())
        with mock.patch.object(sequence, 'inspect_container', return_value={'id': 'a'*64, 'running': True, 'session': 'foreign'}), \
                mock.patch.object(sequence.subprocess, 'run') as docker:
            with self.assertRaisesRegex(RuntimeError, 'another session'):
                self.session.request_pause(child)
        docker.assert_not_called()

    def test_models_must_be_ready_and_are_not_started_by_preflight(self):
        client = mock.Mock()
        client.health.return_value = {'model_id': 'wrong'}
        with mock.patch.object(sequence, 'inspect_container', return_value=None), \
                mock.patch.object(sequence, 'InferenceClient', return_value=client), \
                mock.patch.object(sequence.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(RuntimeError, 'start the policy servers separately'):
                self.session.preflight()
        start.assert_not_called()

    def test_dry_run_does_not_touch_docker_models_or_terminal_input(self):
        with mock.patch.object(sequence.Sequence, 'preflight') as check, \
                mock.patch.object(sequence.Sequence, 'run') as start, \
                mock.patch.object(sequence, 'Console') as console, \
                mock.patch.object(sequence.signal, 'signal'), redirect_stdout(StringIO()) as output:
            self.assertEqual(sequence.main(['--dry-run', '--output', str(self.base/'dry')]), 0)
        check.assert_not_called()
        start.assert_not_called()
        console.assert_not_called()
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['role'] for row in rows], ['continuous-cameras', 'v2', 'v1'])
        self.assertFalse((self.base/'dry').exists())


class ConsoleAndWrapperTests(unittest.TestCase):
    def test_enter_and_eof_are_distinct_controls(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, 'rb') as source:
                console = sequence.Console(source)
                os.write(write_fd, b'\n')
                self.assertTrue(console.enter(0))
                self.assertFalse(console.abort)
                os.close(write_fd)
                write_fd = None
                self.assertFalse(console.enter(0))
                self.assertTrue(console.abort)
        finally:
            if write_fd is not None:
                os.close(write_fd)

    def test_wrapper_adds_non_tty_and_ownership_label_only_for_orchestration(self):
        with tempfile.TemporaryDirectory() as temporary:
            docker = Path(temporary)/'docker'
            docker.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            docker.chmod(0o755)
            env = dict(os.environ, PATH=temporary+':'+os.environ['PATH'])
            env.pop('COSMOS_GATEWAY_NO_TTY', None)
            env.pop('COSMOS_GATEWAY_SESSION_ID', None)
            command = ['bash', str(ROOT/'deploy/5090/run_gateway.sh'), 'execute', '--pause-diagnostic']
            regular = json.loads(subprocess.check_output(command, env=env, text=True))
            self.assertNotIn('-T', regular)
            self.assertNotIn('--label', regular)
            env.update(COSMOS_GATEWAY_NO_TTY='1', COSMOS_GATEWAY_SESSION_ID='owned-session')
            owned = json.loads(subprocess.check_output(command, env=env, text=True))
            self.assertIn('-T', owned)
            self.assertEqual(owned[owned.index('--label')+1], 'cosmos.sequence=owned-session')
            self.assertEqual(owned[-3:], ['gateway', 'execute', '--pause-diagnostic'])


if __name__ == '__main__':
    unittest.main()
