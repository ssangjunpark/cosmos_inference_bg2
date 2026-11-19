#!/usr/bin/env python3
"""Record continuously; Enter finishes v2, then v1, including pause diagnostics.

Model servers and robot bringup must already be running. This process starts and
stops only its camera-recorder container and one execute gateway at a time.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import shlex
import signal
import subprocess
import sys
import time
import uuid

from bg2_bridge.artifact_protocol import ARTIFACT_API, atomic_json
from bg2_bridge.pause_diagnostic import EXTRA_VIDEOS
from bg2_bridge.rpc import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT/'outputs/gateway'
GATEWAY_NAME = 'cosmos-bg2-execute'
RECORDER_NAME = 'cosmos-bg2-cameras'
INSPECT_FORMAT = ('{"id":{{json .Id}},"running":{{json .State.Running}},'
                  '"session":{{json (index .Config.Labels "cosmos.sequence")}}}')


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def inspect_container(name):
    result = subprocess.run(['docker', 'inspect', '--type', 'container', '--format', INSPECT_FORMAT, name],
                            text=True, capture_output=True, timeout=10)
    if result.returncode:
        if 'no such' in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr.strip() or f'Cannot inspect {name}')
    return json.loads(result.stdout)


@dataclass
class Policy:
    model_id: str
    step: int
    profile: str
    url: str


def policies(config_path):
    result = {}
    for entry in read_json(config_path)['models']:
        config = read_json(ROOT/entry['config'])
        result[config['variant']] = Policy(config['variant'], config['step'], config['gateway_profile'],
                                          f'http://127.0.0.1:{entry["port"]}')
    if set(result) != {'v1', 'v2'}:
        raise ValueError('The sequence requires the v1 and v2 paired-service configuration')
    return result


class Console:
    def __init__(self, stream):
        self.fd = stream.fileno()
        self.interrupts = 0
        self.abort = False

    def interrupt(self, *_):
        self.interrupts += 1
        self.abort = True

    def enter(self, timeout=.1):
        readable, _, _ = select.select([self.fd], [], [], timeout)
        if not readable:
            return False
        payload = os.read(self.fd, 4096)
        if not payload:
            self.abort = True
            return False
        # Discard repeated Enter presses while a previous stage is finalizing.
        return b'\n' in payload or b'\r' in payload

    def discard_pending(self):
        while not self.abort and select.select([self.fd], [], [], 0)[0]:
            self.enter(0)


@dataclass
class Child:
    role: str
    name: str
    folder: Path
    process: object
    identity: str | None = None
    interrupts: int = 0


def validate_camera_recording(folder):
    metadata = read_json(folder/'run.json')
    summary = read_json(folder/'recording_summary.json')
    if (metadata.get('episode_state') != 'ended' or summary.get('error') or summary.get('close_errors') or
            any(summary.get('frames', {}).get(key, 0) <= 0 for key in ('head', 'wrist_left', 'wrist'))):
        raise RuntimeError(f'Camera recording did not finalize cleanly: {folder}')
    for key in ('head', 'wrist_left', 'wrist'):
        video, timestamps = folder/'cameras'/(key+'.mkv'), folder/'cameras'/(key+'.timestamps.jsonl')
        if not video.is_file() or video.stat().st_size == 0 or not timestamps.is_file():
            raise RuntimeError(f'Missing finalized camera data: {key} in {folder}')
        with timestamps.open() as stream:
            frames = sum(1 for line in stream if line.strip())
        if frames != summary['frames'][key]:
            raise RuntimeError(f'Camera timestamp/frame count differs: {key} in {folder}')


def validate_gateway_recording(folder):
    validate_camera_recording(folder)
    run = read_json(folder/'run.json')
    diagnostic = read_json(folder/'diagnostics/pause/diagnostic.json')
    if (run.get('pause_diagnostic', {}).get('state') != 'complete' or
            diagnostic.get('state') != 'complete' or diagnostic.get('executed') is not False or
            diagnostic.get('submitted_to_ffw') is not False or
            diagnostic.get('video_layout')!='original_plus_pause_forecast'):
        raise RuntimeError(f'Pause diagnostic did not finish; next policy will not start: {folder}')
    with (folder/'events.jsonl').open() as stream:
        confirmed = any(row.get('event') == 'pause_confirmed' and row.get('epoch') == diagnostic['pause_epoch']
                        for row in (json.loads(line) for line in stream if line.strip()))
    if not confirmed:
        raise RuntimeError('No matching pause acknowledgement; next policy will not start')
    for name in EXTRA_VIDEOS.values():
        path = folder/'imagined'/name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing diagnostic preview: {path}')
    manifest = read_json(folder/'imagined/manifest.json')
    chunks = manifest['chunks']
    valid_cancel=lambda c:(c.get('state')=='cancelled' and c.get('prediction_received') is False and
                          c.get('submitted_to_ffw') is False and c.get('cancellation_proof') in
                          ('client_not_submitted','server_busy_refusal','later_diagnostic_completed'))
    if manifest.get('last_error') or any(not valid_cancel(c) and
            (c.get('state')!='complete' or not c.get('rendered')) for c in chunks):
        raise RuntimeError(f'Normal recordings are incomplete; next policy will not start: {folder}')
    complete=[c for c in chunks if c.get('state')=='complete']
    if complete:
        timeline = read_json(folder/'imagined/timeline.json')
        if timeline.get('partial') or timeline.get('frame_count') != len(complete)*16:
            raise RuntimeError('Normal imagined-video assembly is incomplete')
        for name in ('rollout.mp4', 'head.mp4', 'wrist_left.mp4', 'wrist_right.mp4'):
            if not (folder/'imagined'/name).is_file():
                raise RuntimeError(f'Missing normal imagined preview: {name}')


class Sequence:
    def __init__(self, selected, folder, console, *, instruction, timeout=120., recording_wait=30., startup_timeout=120.):
        self.selected, self.folder, self.console = selected, folder.resolve(), console
        self.folder.relative_to(OUTPUTS.resolve())
        self.instruction, self.timeout = instruction, timeout
        self.recording_wait, self.startup_timeout = recording_wait, startup_timeout
        self.session_id = uuid.uuid4().hex
        self.stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+self.session_id[:8]
        self.recorder = None
        self.gateway = None
        self.children = []
        self.metadata = {'state': 'pending', 'order': ['v2', 'v1'], 'session_id': self.session_id,
                         'started_at_utc': datetime.now(timezone.utc).isoformat(),
                         'policy_servers_managed': False, 'robot_bringup_managed': False, 'runs': {}}

    def update(self, **fields):
        self.metadata.update(fields)
        atomic_json(self.folder/'sequence.json', self.metadata)

    def container_output(self, folder):
        return '/output/'+folder.relative_to(OUTPUTS).as_posix()

    def recorder_command(self):
        folder = self.folder/('continuous-cameras-'+self.stamp)
        return folder, ['bash', 'deploy/5090/record_cameras.sh', '--quiet-status', '--output', self.container_output(folder)]

    def gateway_command(self, model_id):
        policy = self.selected[model_id]
        folder = self.folder/f'execute-yellowstick-{model_id}-step{policy.step}-{self.stamp}'
        return folder, ['bash', 'deploy/5090/run_gateway.sh', 'execute', '--profile', '/config/profile.json',
            '--messages-dir', '/opt/cosmos_gateway/messages', '--server-url', policy.url,
            '--expected-model', model_id, '--timeout', str(self.timeout), '--recording-wait', str(self.recording_wait),
            '--pause-diagnostic', '--quiet-status', '--instruction', self.instruction, '--output', self.container_output(folder)]

    def preflight(self):
        for name in (GATEWAY_NAME, RECORDER_NAME):
            if inspect_container(name) is not None:
                raise RuntimeError(f'{name} already exists; finish that gateway/recorder in its original terminal first')
        for policy in self.selected.values():
            client = InferenceClient(policy.url, 5)
            try:
                health = client.health()
                if (health.get('model_id') != policy.model_id or health.get('checkpoint', {}).get('step') != policy.step or
                        health.get('mode') != 'wam' or health.get('artifacts_api') != ARTIFACT_API or
                        health.get('records_outputs') is not True):
                    raise RuntimeError(f'{policy.model_id} is not ready at {policy.url}; start the policy servers separately')
            finally:
                client.cancel()

    def owned_identity(self, child):
        if child.identity is not None:
            return child.identity
        current = inspect_container(child.name)
        if current is None:
            return None
        if current.get('session') != self.session_id:
            raise RuntimeError(f'{child.name} belongs to another session; leaving it untouched')
        identity = current.get('id', '')
        if not re.fullmatch('[0-9a-f]{64}', identity):
            raise RuntimeError('Invalid owned-container identity')
        child.identity = identity
        return identity

    def start(self, role, name, folder, command, profile):
        env = dict(os.environ, COSMOS_PROFILE='./'+profile, COSMOS_GATEWAY_CONTAINER=name,
                   COSMOS_GATEWAY_NO_TTY='1', COSMOS_GATEWAY_SESSION_ID=self.session_id)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
        child = Child(role, name, folder, process)
        self.children.append(child)
        return child

    def wait_ready(self, child, ready):
        deadline = time.monotonic()+self.startup_timeout
        while True:
            if self.console.abort:
                raise InterruptedError('Sequence cancelled before startup completed')
            if child.process.poll() is not None:
                raise RuntimeError(f'{child.role} exited during startup (code {child.process.returncode})')
            if self.owned_identity(child) and ready():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{child.role} did not become ready')
            time.sleep(.1)

    def request_pause(self, child, *, again=False):
        identity = self.owned_identity(child)
        if identity is None:
            return
        current = inspect_container(identity)
        if current is None or not current.get('running'):
            return
        if current.get('session') != self.session_id:
            raise RuntimeError('Container ownership changed; no signal was sent')
        if child.interrupts and not again:
            return
        subprocess.run(['docker', 'kill', '--signal=SIGINT', identity], check=True,
                       capture_output=True, text=True, timeout=10)
        child.interrupts += 1
        emit('sequence_pause_requested', role=child.role, submitted_to_ffw=False)

    def wait_gateway(self, child):
        self.console.discard_pending()
        print(f'{child.role.upper()} is running. Press Enter to pause, save its diagnostic and finish recordings.', flush=True)
        last_report = 0.
        while child.process.poll() is None:
            if self.recorder.process.poll() is not None:
                raise RuntimeError('Continuous recorder exited unexpectedly; no next policy will start')
            entered = self.console.enter()
            if (entered or self.console.abort) and not child.interrupts:
                self.request_pause(child)
            if self.console.interrupts >= 2 and child.interrupts < 2:
                self.request_pause(child, again=True)
            if child.interrupts and time.monotonic()-last_report >= 5:
                emit('sequence_waiting_for_cleanup', model=child.role)
                last_report = time.monotonic()
        code = child.process.wait()
        if code != 0 or not child.interrupts:
            raise RuntimeError(f'{child.role} exited unexpectedly (code {code}); no next policy will start')
        validate_gateway_recording(child.folder)

    def stop_and_wait(self, child):
        """Wait for application cleanup; never use Docker's short stop timeout."""
        deadline = time.monotonic()+self.startup_timeout
        removal_deadline = None
        while True:
            identity = self.owned_identity(child)
            current = inspect_container(identity) if identity is not None else None
            process_running = child.process.poll() is None
            if current is None and not process_running:
                break
            if current is not None and current.get('running'):
                self.request_pause(child)
                if self.console.interrupts >= 2 and child.interrupts < 2:
                    self.request_pause(child, again=True)
            elif identity is None and process_running and time.monotonic() >= deadline:
                child.process.terminate()
            if current is not None and not current.get('running') and not process_running:
                if removal_deadline is None:
                    removal_deadline = time.monotonic()+10
                if time.monotonic() >= removal_deadline:
                    raise RuntimeError(f'{child.role} container was not removed after cleanup')
            self.console.enter()
        child.process.wait()

    def run(self):
        self.preflight()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.update(state='starting')
        failure = None
        try:
            folder, command = self.recorder_command()
            self.recorder = self.start('continuous-cameras', RECORDER_NAME, folder, command, self.selected['v2'].profile)
            self.update(recorder=str(folder.relative_to(self.folder)))
            self.wait_ready(self.recorder, lambda: all((folder/'cameras'/(key+'.timestamps.jsonl')).exists() and
                (folder/'cameras'/(key+'.timestamps.jsonl')).stat().st_size > 0 for key in ('head', 'wrist_left', 'wrist')))
            for model_id in ('v2', 'v1'):
                if self.console.abort:
                    break
                if self.recorder.process.poll() is not None:
                    raise RuntimeError('Continuous recorder exited; no next policy will start')
                folder, command = self.gateway_command(model_id)
                self.gateway = self.start(model_id, GATEWAY_NAME, folder, command, self.selected[model_id].profile)
                self.metadata['runs'][model_id] = str(folder.relative_to(self.folder))
                self.update(state='running', active_model=model_id)
                self.wait_ready(self.gateway, lambda: (folder/'events.jsonl').is_file())
                self.wait_gateway(self.gateway)
                self.stop_and_wait(self.gateway)
                emit('sequence_policy_saved', model=model_id, directory=str(folder))
                self.gateway = None
            self.update(state='finishing')
        except InterruptedError as error:
            if not self.console.abort:
                failure = error
        except Exception as error:
            failure = error
        finally:
            errors = []
            # Finish the gateway before ending the session-wide camera recording.
            for child in reversed(self.children):
                try:
                    self.stop_and_wait(child)
                except Exception as error:
                    errors.append(str(error))
            if failure is None and errors:
                failure = RuntimeError('; '.join(errors))
            if failure is None and self.recorder is not None:
                try:
                    validate_camera_recording(self.recorder.folder)
                except Exception as error:
                    failure = error
            state = 'failed' if failure is not None else 'cancelled' if self.console.abort else 'complete'
            self.update(state=state, active_model=None, error=str(failure) if failure else None, cleanup_errors=errors,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        if failure is not None:
            raise failure
        emit('sequence_saved', state=self.metadata['state'], directory=str(self.folder))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/5090/yellowstick_pair.json')
    parser.add_argument('--instruction', default='Move the yellow stick into the basket.')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--recording-wait', type=float, default=30)
    parser.add_argument('--startup-timeout', type=float, default=120)
    parser.add_argument('--output', type=Path, help='New session directory under outputs/gateway/')
    parser.add_argument('--dry-run', action='store_true', help='Print recorder/v2/v1 commands without starting or contacting anything')
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 120 or args.recording_wait < 0 or args.startup_timeout <= 0 or not args.instruction.strip():
        parser.error('Invalid instruction or timeout')
    folder = args.output or OUTPUTS/('sequence-yellowstick-v2-v1-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    console = DryConsole() if args.dry_run else Console(sys.stdin)
    previous = {sig: signal.signal(sig, console.interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        sequence = Sequence(policies(args.config), folder, console, instruction=args.instruction,
                            timeout=args.timeout, recording_wait=args.recording_wait, startup_timeout=args.startup_timeout)
        if args.dry_run:
            for role, (_, command) in [('continuous-cameras', sequence.recorder_command()),
                                       ('v2', sequence.gateway_command('v2')), ('v1', sequence.gateway_command('v1'))]:
                policy = sequence.selected['v2' if role == 'continuous-cameras' else role]
                emit('sequence_plan', role=role, profile=policy.profile, command=shlex.join(command))
            return 0
        sequence.run()
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        emit('sequence_failed', error=str(error))
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class DryConsole:
    """Dry-run placeholder that never touches terminal input."""
    abort = False
    interrupts = 0
    def interrupt(self, *_):
        self.abort = True


if __name__ == '__main__':
    raise SystemExit(main())
