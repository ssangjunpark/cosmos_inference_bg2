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

from bg2_bridge.core import require
from bg2_bridge.chunks import SCHEMA
from bg2_bridge.rpc import MAX_OBSERVATION, decode_observation, identifier
from bg2_inference import JOINT_NAMES, load_observation
from bg2_bridge.recording import save_model_result


class PolicyService:
    def __init__(self, factory, sampling, warmup, warmup_runs=2, output_dir=None):
        require(warmup_runs >= 1, 'At least one explicit warm-up is required')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cosmos-gpu')
        self.gate = threading.Lock()
        self.sampling = dict(sampling)
        self.durations = deque(maxlen=50)
        self.calls = 0
        self.output_dir=Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:self.output_dir.mkdir(parents=True,exist_ok=True)
        try:
            # Load, warm, and predict on the same dedicated thread. No CUDA request
            # concurrency and no new per-request GPU thread/handle warmup.
            self.policy = self.pool.submit(factory).result()
            self.warmup_seconds = []
            for _ in range(warmup_runs):
                start = time.monotonic()
                self.pool.submit(self.policy.predict, warmup, **self.sampling).result()
                self.warmup_seconds.append(time.monotonic() - start)
            self.durations.extend(self.warmup_seconds[1:] or self.warmup_seconds)
        except Exception:
            self.pool.shutdown(wait=True)
            raise

    def health(self):
        values = list(self.durations)
        return {'ok': True, 'warmed': True, 'schema': 'cosmos_inference_service_v1', 'joint_names': JOINT_NAMES,
                'future_steps': 16, 'frame_interval_ns': 50_000_000, 'loading_seconds': self.policy.loading_seconds,
                'warmup_seconds': self.warmup_seconds, 'latency_budget_ns': int(max(values) * 1e9),
                'median_prediction_seconds': statistics.median(values), 'prediction_calls': self.calls,
                'sampling': self.sampling, 'mode': 'policy', 'records_outputs': self.output_dir is not None}

    def predict(self, observation, epoch, request_id):
        start = time.monotonic()
        result = self.pool.submit(self.policy.predict, observation, **self.sampling).result()
        elapsed = time.monotonic() - start
        self.durations.append(elapsed)
        self.calls += 1
        response = {'ok': True, 'schema': SCHEMA, 'epoch': epoch, 'request_id': request_id,
                'source_time_ns': int(observation['source_time_ns']), 'joint_names': JOINT_NAMES,
                'units': ['rad'] * 8, 'positions': result['positions'].tolist(),
                'timings': {**result['timings'], 'service_prediction_seconds': elapsed}}
        if self.output_dir is not None:
            response['artifacts']=save_model_result(self.output_dir,observation,result,response,self.sampling)
        return response

    def close(self):
        self.pool.shutdown(wait=True)


def make_server(address, service):
    class Handler(BaseHTTPRequestHandler):
        timeout = 10

        def log_message(self, *_):
            pass  # Do not log observation payloads.

        def reply(self, status, data):
            payload = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
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
                payload = self.rfile.read(size)
                require(len(payload) == size, 'Incomplete observation body')
                observation = decode_observation(payload)
            except Exception as error:
                service.gate.release()
                self.reply(400, {'ok': False, 'error': str(error)})
                return
            try:
                result = service.predict(observation, epoch, request_id)
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
    args = parser.parse_args()
    from policy_runtime import NativePolicy
    sampling = {k: getattr(args, k) for k in ['seed', 'steps', 'guidance', 'shift', 'guidance_interval']}
    service = PolicyService(lambda: NativePolicy(args.checkpoint, args.tokenizer, args.vae, args.adapter_dir),
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
