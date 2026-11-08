"""WAM service compatibility through the existing client; no robot or GPU needed."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bg2_bridge.rpc import InferenceClient, MAX_REPLY
from inference_server import PolicyService, make_server
from test_network_inference import FakePolicy, observation


class FakeWAM(FakePolicy):
    mode = 'wam'
    def predict(self, obs, **kwargs):
        result = super().predict(obs, **kwargs)
        positions = np.full((16, 8), .25, np.float32)
        positions[:, -1] = -.05
        self.raw_positions = positions
        frames = np.zeros((17, 640, 512, 3), np.uint8)
        frames[:, :384, :, 0] = 200
        frames[:, 384:, :256, 1] = 180
        frames[:, 384:, 256:, 2] = 160
        return {**result, 'mode': 'wam', 'positions': positions,
                'timings': {**result['timings'], 'video_decoding_seconds': .001,
                    'peak_gpu_allocated_bytes': 0, 'peak_gpu_reserved_bytes': 0},
                'normalized': np.zeros((17, 8), np.float32), 'imagined_video': frames,
                'vision': [np.zeros((1, 48, 5, 40, 32), np.float32)],
                'prompt': str(obs['instruction']), 'runtime_metadata': {'mode': 'wam'}}


class WAMServiceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch required by offline CLI')
    def test_offline_cli_reuses_wam_runtime_and_retains_raw_negative_gripper(self):
        from types import SimpleNamespace
        import wam_infer
        import wam_runtime
        root=Path('/mnt/sata1/RobotEscapeCkpt/cosmos3_wam_room20k_20261105')
        if not (root/'edge/020000/READY.json').exists():self.skipTest('Local WAM checkpoint unavailable')
        fake=FakeWAM()
        fake.provenance={};fake.config_overrides=[]
        fake.formatter=SimpleNamespace(build_policy_prompt=lambda instruction:json.dumps({'instruction':instruction}))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);sample=path/'sample.npz'
            np.savez(sample,**observation())
            output=path/'output'
            args=['wam_infer.py','--checkpoint',str(root/'edge/020000'),
                  '--experiment-contract',str(root/'connection_test/experiment_contract.json'),
                  '--observation',str(sample),'--tokenizer',str(path),'--vae',str(path/'unused.pth'),
                  '--output',str(output)]
            with mock.patch.object(sys,'argv',args),mock.patch.object(wam_runtime,'NativeWAM',return_value=fake), \
                    mock.patch('torch.cuda.get_device_name',return_value='CPU fake'), \
                    mock.patch('bg2_wam.save_rollout',return_value={'frame_count':17,'fps':20}):
                wam_infer.main()
            self.assertTrue((np.load(output/'joint_positions.npy')[:,-1]<0).all())
            self.assertEqual(json.loads((output/'request.json').read_text())['mode'],'wam')
            self.assertEqual(json.loads((output/'outputs.json').read_text())['actions_shape'],[16,8])

    def test_existing_client_gets_same_chunk_and_completed_local_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PolicyService(FakeWAM, {'seed': 0}, observation(), 1, directory)
            server = make_server(('127.0.0.1', 0), service)
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
            thread.start()
            try:
                client = InferenceClient(f'http://127.0.0.1:{server.server_address[1]}', timeout=30)
                self.assertEqual(client.health()['mode'], 'wam')
                self.assertTrue(client.health()['imagined_future_rollout'])
                warmups=list((Path(directory)/'warmup').glob('*/*/imagined_future.mp4'))
                self.assertEqual(len(warmups),1)
                from bg2_bridge.chunks import ChunkExecutor, EXECUTION_CONTRACT
                from test_ffw_companion import ready, feedback
                startup, measured = ready()
                executor = ChunkExecutor(startup)
                executor.tick(measured, measured.source_ns)
                obs = observation(measured.source_ns)
                epoch, ident = startup.epoch, uuid.uuid4().hex
                result = client.predict(obs, epoch, ident)
                self.assertEqual(len(json.dumps(result).encode()) < MAX_REPLY, True)
                self.assertEqual(result['mode'], 'wam')
                targets = np.asarray(result['positions'])
                np.testing.assert_array_equal(targets[:, :7], np.full((16, 7), .25))
                np.testing.assert_array_equal(targets[:, 7], np.zeros(16))
                self.assertTrue((service.policy.raw_positions[:, 7] < 0).all())
                folder = Path(directory) / result['artifacts']['directory']
                self.assertTrue((np.load(folder / 'raw_joint_positions.npy')[:, 7] < 0).all())
                np.testing.assert_array_equal(np.load(folder / 'joint_positions.npy'), targets)
                metadata = json.loads((folder / 'model_output.json').read_text())
                self.assertEqual(metadata['mode'], 'wam')
                self.assertTrue(metadata['imagined_future_rollout'])
                for name in result['artifacts']['videos'].values():
                    self.assertTrue((folder / name).is_file())
                self.assertEqual(np.load(folder / 'decoded_rollout.npy').shape, (17, 640, 512, 3))
                # Same extras and 16 points pass the actual companion's pure executor.
                now = measured.source_ns + 20_000_000
                chunk = {**result, 'execution_contract': EXECUTION_CONTRACT, 'cycle': 0}
                commands = executor.accept(chunk, feedback(startup.profile, now, measured.positions), now)
                self.assertEqual([p['positions'] for p in commands['arm_right']['message']['points']], result['positions'])
                end = executor.execution_end_ns
                actual = {**measured.positions, **dict(zip(result['joint_names'], result['positions'][-1]))}
                hold = executor.tick(feedback(startup.profile, end, actual), end)
                self.assertEqual(hold['arm_right']['message']['points'][0]['positions'], result['positions'][-1])
                self.assertTrue(executor.status(end)['needs_chunk'])
                self.assertEqual(len(set(service.policy.threads)), 1)
                self.assertEqual(client.health()['prediction_calls'], 1)
            finally:
                server.shutdown(); server.server_close(); thread.join(2); service.close()

    def test_failed_video_save_does_not_return_an_executable_chunk_and_releases_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            service = PolicyService(FakeWAM, {}, observation(), 1, directory)
            server = make_server(('127.0.0.1', 0), service)
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
            thread.start()
            try:
                client = InferenceClient(f'http://127.0.0.1:{server.server_address[1]}', timeout=30)
                with mock.patch('bg2_wam.save_rollout', side_effect=RuntimeError('encoding failed')):
                    with self.assertRaisesRegex(ValueError, 'encoding failed'):
                        client.predict(observation(), uuid.uuid4().hex, uuid.uuid4().hex)
                self.assertTrue(service.gate.acquire(blocking=False)); service.gate.release()
                self.assertEqual(client.health()['prediction_calls'], 0)
            finally:
                server.shutdown(); server.server_close(); thread.join(2); service.close()

    def test_wam_serve_dry_run_selects_new_root_readonly_and_preserves_rpc_port(self):
        args = ['python3', 'scripts/cosmos3.py', 'serve', '--mode', 'wam', '--variant', 'edge', '--gpu', '4',
                '--warmup-observation', 'inputs/recorded_observation.npz',
                '--tokenizer', 'assets/nvidia--Cosmos3-Edge-344d602b128d1bbdacb43b08d0a3626f46343e29',
                '--vae', 'assets/Wan-AI--Wan2.2-TI2V-5B-921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth',
                '--port', '19018', '--dry-run']
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn('cosmos3_wam_room20k_20261105/edge/020000', result.stdout)
        self.assertIn('target=/checkpoint,readonly', result.stdout)
        self.assertIn('127.0.0.1:19018:8008', result.stdout)
        self.assertIn('--experiment-contract /input/experiment_contract.json', result.stdout)
        self.assertIn('cosmos-bg2-wam-edge-server', result.stdout)
        self.assertIn('--guidance-interval 960.0 1001.0', result.stdout)


if __name__ == '__main__':
    unittest.main()
