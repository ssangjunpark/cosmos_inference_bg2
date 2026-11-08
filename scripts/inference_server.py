#!/usr/bin/env python3
"""Persistent Cosmos HTTP service. Publish its port on localhost and use an SSH tunnel."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import signal
import threading
import time

import numpy as np

from bg2_bridge.core import require
from bg2_bridge.chunks import SCHEMA
from bg2_bridge.rpc import MAX_OBSERVATION, decode_observation, identifier
from bg2_inference import JOINT_NAMES, load_observation
from bg2_bridge.recording import save_model_result
from bg2_bridge.artifact_protocol import ARTIFACT_API, run_name


class PolicyService:
    def __init__(self, factory, sampling, warmup, warmup_runs=2, output_dir=None):
        require(warmup_runs >= 1, 'At least one explicit warm-up is required')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cosmos-gpu')
        self.gate = threading.Lock()
        self.sampling = dict(sampling)
        self.durations = deque(maxlen=50)
        self.calls = 0
        self.mode = 'policy'
        self.recordings = None
        self.output_dir=Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:self.output_dir.mkdir(parents=True,exist_ok=True)
        try:
            # Load, warm, and predict on the same dedicated thread. No CUDA request
            # concurrency and no new per-request GPU thread/handle warmup.
            self.policy = self.pool.submit(factory).result()
            self.mode = getattr(self.policy, 'mode', 'policy')
            self.warmup_seconds = []
            for _ in range(warmup_runs):
                start = time.monotonic()
                result = self.pool.submit(self.policy.predict, warmup, **self.sampling).result()
                self.warmup_seconds.append(time.monotonic() - start)
            self.durations.extend(self.warmup_seconds[1:] or self.warmup_seconds)
            if self.mode == 'wam' and self.output_dir is not None:
                from model_recording import ModelRecordings
                self.recordings = ModelRecordings(self.output_dir)
        except Exception:
            self.pool.shutdown(wait=True)
            raise

    def health(self):
        values = list(self.durations)
        return {'ok': True, 'warmed': True, 'schema': 'cosmos_inference_service_v1', 'joint_names': JOINT_NAMES,
                'future_steps': 16, 'frame_interval_ns': 50_000_000, 'loading_seconds': self.policy.loading_seconds,
                'warmup_seconds': self.warmup_seconds, 'latency_budget_ns': int(max(values) * 1e9),
                'median_prediction_seconds': statistics.median(values), 'prediction_calls': self.calls,
                'sampling': self.sampling, 'mode': self.mode, 'records_outputs': self.output_dir is not None,
                'imagined_future_rollout': self.mode == 'wam', 'video_frames': 17 if self.mode == 'wam' else 1,
                'artifacts_api': ARTIFACT_API if self.recordings is not None else None,
                'recording_error': self.recordings.last_error if self.recordings is not None else None}

    def predict(self, observation, epoch, request_id, run_id=None):
        start = time.monotonic()
        result = self.pool.submit(self.policy.predict, observation, **self.sampling).result()
        elapsed = time.monotonic() - start
        response = self.response(observation, result, epoch, request_id, elapsed)
        if self.output_dir is not None:
            if self.recordings is not None:
                try:
                    response['artifacts'] = self.recordings.submit(observation,result,response,self.sampling,run_id or epoch)
                except Exception as error:
                    self.recordings.last_error = str(error)
                    response['artifacts'] = {'state':'failed','error':str(error)}
            else:
                response['artifacts']=save_model_result(self.output_dir,observation,result,response,self.sampling)
        self.durations.append(time.monotonic() - start)
        self.calls += 1
        return response

    def response(self, observation, result, epoch, request_id, elapsed):
        positions = np.array(result['positions'], copy=True)
        if self.mode == 'wam':
            # Existing operator-requested command rule. WAM runtime/raw files stay untouched.
            gripper = JOINT_NAMES.index('gripper_r_joint1')
            positions[:, gripper] = np.maximum(positions[:, gripper], 0.)
        return {'ok': True, 'schema': SCHEMA, 'epoch': epoch, 'request_id': request_id,
                'mode': self.mode,
                'source_time_ns': int(observation['source_time_ns']), 'joint_names': JOINT_NAMES,
                'units': ['rad'] * 8, 'positions': positions.tolist(),
                'timings': {**result['timings'], 'service_prediction_seconds': elapsed}}

    def close(self):
        self.pool.shutdown(wait=True)
        if self.recordings is not None:self.recordings.close()


def make_server(address, service):
    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def log_message(self, *_):
            pass  # Do not log observation payloads.

        def reply(self, status, data):
            payload = json.dumps(data, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.startswith('/artifact-lookup/') and service.recordings is not None:
                try:
                    parts=self.path.split('/')[2:]
                    if len(parts)!=2:raise ValueError('Invalid recording lookup')
                    self.reply(200,service.recordings.find(*parts))
                except (ValueError,FileNotFoundError):
                    self.reply(404,{'ok':False,'error':'Saved request not found'})
                return
            if self.path.startswith('/artifacts/') and service.recordings is not None:
                try:
                    parts = self.path.split('/')[2:]
                    if len(parts) == 3:
                        self.reply(200, service.recordings.status(*parts))
                    elif len(parts) == 4:
                        path = service.recordings.file(*parts)
                        self.send_response(200)
                        self.send_header('Content-Type','application/octet-stream')
                        self.send_header('Content-Length',str(path.stat().st_size))
                        self.end_headers()
                        with path.open('rb') as stream:
                            while block := stream.read(64*1024):self.wfile.write(block)
                    else:
                        raise ValueError('Unknown recording endpoint')
                except (FileNotFoundError, ValueError):
                    self.reply(404,{'ok':False,'error':'Recording not available'})
                except (BrokenPipeError,ConnectionResetError):
                    pass
                return
            if self.path != '/health':
                self.reply(404, {'ok': False, 'error': 'Unknown endpoint'})
                return
            self.reply(200, service.health())

        def do_POST(self):
            if self.path != '/predict':
                self.reply(404, {'ok': False, 'error': 'Unknown endpoint'})
                return
            if not service.gate.acquire(blocking=False):
                self.reply(503, {'ok': False, 'error': 'Model is busy; no observation queue is maintained'})
                return
            try:
                require(self.headers.get('Content-Type') == 'application/x-npz', 'Expected a pickle-free observation NPZ')
                require(self.headers.get('Transfer-Encoding') is None, 'Chunked HTTP bodies are not supported')
                size = int(self.headers.get('Content-Length', '0'))
                require(0 < size <= MAX_OBSERVATION, 'Invalid observation body length')
                epoch = identifier(self.headers.get('X-Session-Epoch'))
                request_id = identifier(self.headers.get('X-Request-ID'))
                run_id = self.headers.get('X-Run-ID')
                if run_id is not None:run_name(run_id)
                payload = self.rfile.read(size)
                require(len(payload) == size, 'Incomplete observation body')
                observation = decode_observation(payload)
            except Exception as error:
                service.gate.release()
                self.reply(400, {'ok': False, 'error': str(error)})
                return
            try:
                result = service.predict(observation, epoch, request_id, run_id)
                self.reply(200, result)
            except Exception as error:
                self.reply(500, {'ok': False, 'error': str(error)})
            finally:
                service.gate.release()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 2

    return Server(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'tokenizer', 'vae', 'warmup-observation']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, default=Path('/adapter'))
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8008)
    parser.add_argument('--warmup-runs', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('/results'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--guidance', type=float, default=3.)
    parser.add_argument('--shift', type=float, default=5.)
    parser.add_argument('--guidance-interval', type=float, nargs=2)
    parser.add_argument('--mode', choices=('policy', 'wam'), default='policy')
    parser.add_argument('--experiment-contract', type=Path)
    parser.add_argument('--step', type=int, default=20000)
    args = parser.parse_args()
    if args.mode == 'wam':
        if args.experiment_contract is None:
            parser.error('--mode wam requires --experiment-contract')
        from wam_runtime import NativeWAM
        factory = lambda: NativeWAM(args.checkpoint, args.tokenizer, args.vae, args.adapter_dir,
                                    args.experiment_contract, step=args.step)
    else:
        from policy_runtime import NativePolicy
        factory = lambda: NativePolicy(args.checkpoint, args.tokenizer, args.vae, args.adapter_dir)
    sampling = {k: getattr(args, k) for k in ['seed', 'steps', 'guidance', 'shift', 'guidance_interval']}
    service = PolicyService(factory,
                            sampling, load_observation(args.warmup_observation), args.warmup_runs, args.output)
    server = make_server((args.bind, args.port), service)
    print(json.dumps({'event': 'ready', **service.health()}), flush=True)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
