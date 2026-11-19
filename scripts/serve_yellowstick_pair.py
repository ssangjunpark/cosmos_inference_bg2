#!/usr/bin/env python3
"""Start and supervise two resident Edge WAM services on one GPU.

Load/warm serially, keep both resident, and serve them on separate localhost ports.
This launcher never opens robot transport or starts a gateway/recorder.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import cosmos3

ROOT = cosmos3.ROOT


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


@dataclass
class ModelPlan:
    model_id: str
    step: int
    port: int
    name: str
    output: Path
    command: list[str]

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}'


def model_plans(config, *, gpu=None, dry_run=True):
    if config['schema'] != 1 or len(config['models']) != 2:
        raise ValueError('Expected two models in the paired-service config')
    gpu = str(gpu if gpu is not None else config['gpu'])
    plans = []
    for entry in config['models']:
        deployment = json.loads((ROOT / entry['config']).read_text())
        prepared = ROOT / deployment['prepared_root']
        source = Path(deployment['source_root']) / deployment['source_checkpoint']
        output = ROOT / entry['output']
        args = argparse.Namespace(mode='wam', variant=deployment['model_tier'], step=deployment['step'],
            gpu=gpu, checkpoint_root=prepared, model_dir=source/'model',
            experiment_contract=prepared/'experiment_contract.json', adapter_dir=prepared/'adapter',
            tokenizer=ROOT/'assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29',
            vae=ROOT/'assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth',
            warmup_observation=ROOT/'inputs/recorded_observation.npz', warmup_runs=2,
            port=entry['port'], container_name=entry['container_name'], output=output,
            image='base', seed=0, steps=4, guidance=3., shift=5., guidance_interval=None,
            guidance_all_steps=False, allow_busy_gpu=True, dry_run=dry_run)
        if deployment['model_tier'] != 'edge':
            raise ValueError('This paired deployment expects Edge checkpoints')
        command = cosmos3.server_command(args)
        plans.append(ModelPlan(deployment['variant'], deployment['step'], entry['port'],
                               entry['container_name'], output.resolve(), command))
    for field in ('model_id', 'port', 'name', 'output'):
        if len({getattr(plan, field) for plan in plans}) != len(plans):
            raise ValueError(f'Paired services must have distinct {field} values')
    return plans


def check_environment(plans, gpu, minimum_free_gib):
    if not 0 < minimum_free_gib:
        raise ValueError('minimum_free_vram_gib must be positive')
    existing = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}'],
        check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()
    for plan in plans:
        if plan.name in existing:
            raise RuntimeError(f'Container {plan.name} already exists; no existing container was stopped')
        with socket.socket() as listener:
            try:
                listener.bind(('127.0.0.1', plan.port))
            except OSError as error:
                raise RuntimeError(f'Port {plan.port} is occupied; stop the old model in its original terminal first') from error
    free_mib = int(subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=15).stdout.strip())
    if free_mib < minimum_free_gib * 1024:
        raise RuntimeError(f'GPU {gpu} has {free_mib / 1024:.2f} GiB free; this paired config requires {minimum_free_gib:g} GiB')
    emit('gpu_memory', gpu=gpu, free_gib=free_mib / 1024)


def health(url):
    with urlopen(url + '/health', timeout=1) as response:
        return json.loads(response.read(128 * 1024))


def wait_ready(plan, process, stopped, timeout):
    deadline = time.monotonic() + timeout
    while not stopped.is_set():
        if process.poll() is not None:
            raise RuntimeError(f'{plan.model_id} exited before it became ready (code {process.returncode})')
        try:
            status = health(plan.url)
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            status = None
        if status and status.get('ok') and status.get('warmed'):
            if (status.get('model_id') != plan.model_id or
                    status.get('checkpoint', {}).get('step') != plan.step):
                raise RuntimeError(f'{plan.url} reports a different checkpoint than {plan.model_id} step {plan.step}')
            if status.get('mode') != 'wam' or status.get('future_steps') != 16 or not status.get('records_outputs'):
                raise RuntimeError(f'{plan.model_id} reports an incompatible inference/recording contract')
            emit('model_ready', model_id=plan.model_id, step=plan.step, url=plan.url)
            return True
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{plan.model_id} did not become ready within {timeout:g} seconds')
        stopped.wait(.5)
    return False


def stop_owned(process, cidfile):
    """Stop only the exact container ID created by this Docker run invocation."""
    def stop_container():
        if not cidfile.exists():
            return False
        identity = cidfile.read_text().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', identity):
            raise RuntimeError(f'Invalid owned-container receipt: {cidfile}')
        result = subprocess.run(['docker', 'stop', '--timeout', '20', identity],
            capture_output=True, text=True, timeout=30)
        if result.returncode and 'no such container' not in result.stderr.lower():
            raise RuntimeError(result.stderr.strip() or 'Could not stop owned model container')
        return True

    stopped_by_id = False
    try:
        stopped_by_id = stop_container()
    finally:
        if process.poll() is None and not stopped_by_id:
            process.terminate()
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        # The CID can appear during cancellation of Docker's create/start phase.
        if not stopped_by_id:
            stop_container()


def supervise(plans, stopped, startup_timeout):
    runtime = ROOT/'.runtime/5090'
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='yellowstick-pair-', dir=runtime) as temporary:
        owned = []
        try:
            for plan in plans:
                if stopped.is_set():
                    return
                cidfile = Path(temporary)/(plan.model_id + '.cid')
                command = plan.command[:2] + ['--cidfile', str(cidfile)] + plan.command[2:]
                emit('model_starting', model_id=plan.model_id, step=plan.step, url=plan.url)
                process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
                owned.append((process, cidfile))
                if not wait_ready(plan, process, stopped, startup_timeout):
                    return
            emit('dual_ready', models={plan.model_id: {'url': plan.url, 'step': plan.step} for plan in plans})
            while not stopped.wait(.5):
                for plan, (process, _) in zip(plans, owned):
                    if process.poll() is not None:
                        raise RuntimeError(f'{plan.model_id} exited unexpectedly (code {process.returncode})')
        finally:
            errors = []
            for process, cidfile in reversed(owned):
                try:
                    stop_owned(process, cidfile)
                except Exception as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('Model cleanup failed: ' + '; '.join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'deploy/5090/yellowstick_pair.json')
    parser.add_argument('--gpu', help='Physical GPU index; defaults to the paired config')
    parser.add_argument('--startup-timeout', type=float, default=600, help='Maximum warmup wait per model, in seconds')
    parser.add_argument('--dry-run', action='store_true', help='Validate metadata and print commands; no Docker/GPU/network calls')
    args = parser.parse_args(argv)
    stopped = threading.Event()
    previous = {signum: signal.signal(signum, lambda *_: stopped.set()) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.startup_timeout <= 0:
            raise ValueError('--startup-timeout must be positive')
        config = json.loads(args.config.read_text())
        gpu = str(args.gpu if args.gpu is not None else config['gpu'])
        # Validate both plans before creating any containers or output directories.
        plans = model_plans(config, gpu=gpu, dry_run=True)
        if args.dry_run:
            for plan in plans:
                emit('model_plan', model_id=plan.model_id, step=plan.step, url=plan.url,
                     output=str(plan.output), command=shlex.join(plan.command))
            emit('dual_plan_valid', model_started=False)
            return 0
        check_environment(plans, gpu, config['minimum_free_vram_gib'])
        plans = model_plans(config, gpu=gpu, dry_run=False)
        supervise(plans, stopped, args.startup_timeout)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        emit('dual_failed', error=str(error))
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
