"""Authenticated localhost RPC checks with a CPU fake policy; no ROS or robot connection."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import http.client
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
import uuid
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from bg2_inference import FORMAT, JOINT_NAMES
from bg2_bridge.core import BridgeError
from bg2_bridge.rpc import InferenceClient, decode_observation, encode_observation, token_from_file, MAX_OBSERVATION
from inference_server import PolicyService, make_server

TOKEN = 'local-test-token-with-at-least-24-characters'


def observation(timestamp=10_000_000_000):
    return {'head': np.zeros((384,512,3),dtype=np.uint8), 'wrist':np.zeros((256,256,3),dtype=np.uint8),
            'wrist_left':np.zeros((256,256,3),dtype=np.uint8), 'joint_position':np.zeros(8,dtype=np.float32),
            'joint_names':np.asarray(JOINT_NAMES),'source_time_ns':np.asarray(timestamp,dtype=np.int64),
            'instruction':np.asarray('Pick up the garbage truck.'),'format':np.asarray(FORMAT)}


class FakePolicy:
    loading_seconds = .01
    def __init__(self):
        self.threads = [threading.get_ident()]
        self.calls = 0
        self.block = None
    def predict(self, obs, **kwargs):
        self.threads.append(threading.get_ident());self.calls += 1
        if self.block is not None:self.block.wait(3)
        return {'positions':np.tile(obs['joint_position'],(16,1)), 'timings':{'inference_seconds':.001}}


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.service=PolicyService(FakePolicy,{},observation(),2)
        self.server=make_server(('127.0.0.1',0),self.service,TOKEN)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start()
        self.port=self.server.server_address[1]
        self.client=InferenceClient(f'http://127.0.0.1:{self.port}',TOKEN,timeout=5)
    def tearDown(self):
        if self.service.policy.block is not None:self.service.policy.block.set()
        self.server.shutdown();self.server.server_close();self.thread.join(2);self.service.close()

    def test_warmup_and_repeated_requests_reuse_one_loaded_model_and_gpu_thread(self):
        health=self.client.health();self.assertTrue(health['warmed'])
        for _ in range(3):
            epoch,identifier=uuid.uuid4().hex,uuid.uuid4().hex
            result=self.client.predict(observation(),epoch,identifier)
            self.assertEqual(result['source_time_ns'],10_000_000_000)
            self.assertEqual(result['epoch'],epoch)
            self.assertEqual(np.asarray(result['positions']).shape,(16,8))
        self.assertEqual(self.service.policy.calls,5)
        self.assertEqual(len(set(self.service.policy.threads)),1)
        self.assertEqual(self.client.health()['prediction_calls'],3)

    def test_bad_authentication_is_rejected(self):
        with self.assertRaisesRegex(BridgeError,'Authentication'):
            InferenceClient(f'http://127.0.0.1:{self.port}','wrong').health()

    def test_second_inference_request_is_rejected_instead_of_queued(self):
        gate=threading.Event();self.service.policy.block=gate
        with ThreadPoolExecutor(max_workers=1) as pool:
            first=pool.submit(self.client.predict,observation(),uuid.uuid4().hex,uuid.uuid4().hex)
            deadline=time.monotonic()+2
            while not self.service.gate.locked() and time.monotonic()<deadline:time.sleep(.005)
            self.assertTrue(self.service.gate.locked())
            self.assertTrue(self.client.health()['warmed'])
            with self.assertRaisesRegex(BridgeError,'busy'):
                self.client.predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)
            gate.set();self.assertTrue(first.result(3)['ok'])

    def test_bad_archive_and_identity_release_the_busy_slot(self):
        connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=3)
        connection.request('POST','/predict',body=b'bad',headers={'Authorization':'Bearer '+TOKEN,
            'Content-Type':'application/x-npz','X-Session-Epoch':'../escape','X-Request-ID':uuid.uuid4().hex})
        response=connection.getresponse();self.assertEqual(response.status,400);response.read();connection.close()
        self.assertTrue(self.client.predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)['ok'])


class WireValidationTests(unittest.TestCase):
    def test_pickle_free_roundtrip_preserves_rgb_and_original_timestamp(self):
        obs=observation(1234567890123456789)
        obs['head'][1,2]=[1,2,3]
        decoded=decode_observation(encode_observation(obs))
        self.assertEqual(int(decoded['source_time_ns']),1234567890123456789)
        np.testing.assert_array_equal(decoded['head'],obs['head'])

    def test_archive_extra_entries_and_zip_bomb_are_rejected(self):
        for name,value in [('unexpected.npy',b'x'),('head.npy',b'x'*(MAX_OBSERVATION+1))]:
            buffer=BytesIO()
            with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for key in observation():archive.writestr(key+'.npy', value if key+'.npy'==name else b'x')
                if name=='unexpected.npy':archive.writestr(name,value)
            with self.assertRaises((ValueError,BridgeError)):decode_observation(buffer.getvalue())

    def test_object_arrays_and_wrong_joint_order_are_rejected(self):
        for change in [{'instruction':np.array({},dtype=object)}, {'joint_names':np.asarray(JOINT_NAMES[::-1])}]:
            obs={**observation(),**change}
            with self.assertRaises(ValueError):decode_observation(encode_observation(obs))

    def test_plain_remote_urls_and_embedded_credentials_are_rejected(self):
        for url in ['http://192.168.0.14:8008','http://user:password@127.0.0.1:8008','http://127.0.0.1:8008/path']:
            with self.assertRaises(BridgeError):InferenceClient(url,TOKEN)

    def test_token_permissions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'token';path.write_text(TOKEN);path.chmod(0o600)
            self.assertEqual(token_from_file(path),TOKEN)
            path.chmod(0o644)
            with self.assertRaises(BridgeError):token_from_file(path)
            path.chmod(0o600);link=Path(directory)/'link';link.symlink_to(path)
            with self.assertRaises(BridgeError):token_from_file(link)


class ServeLauncherTests(unittest.TestCase):
    def test_dry_run_is_loopback_only_and_does_not_create_credentials(self):
        from argparse import Namespace
        from unittest import mock
        import cosmos3
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokenizer=root/'tokenizer';tokenizer.mkdir()
            (tokenizer/'tokenizer_config.json').write_text('{}')
            vae=root/'vae.pth';vae.write_bytes(b'vae')
            sample=root/'sample.npz';sample.write_bytes(b'sample')
            secret=root/'new-token'
            args=Namespace(checkpoint_root=root,variant='edge',step=20000,tokenizer=tokenizer,vae=vae,
                           warmup_observation=sample,port=8008,warmup_runs=2,token_file=secret,dry_run=True,
                           gpu='5',adapter_dir=cosmos3.ROOT/'adapters',image='base',seed=0,steps=4,
                           guidance=3.,shift=5.,guidance_interval=None)
            with mock.patch.object(cosmos3,'published_checkpoint',return_value=root), mock.patch.object(cosmos3,'checked') as execute:
                cosmos3.serve(args)
            command=execute.call_args.args[0]
            self.assertIn('127.0.0.1:8008:8008',command)
            self.assertIn('--network=bridge',command)
            self.assertIn('--read-only',command)
            self.assertIn('--gpus=device=5',command)
            self.assertNotIn('--privileged',command)
            self.assertFalse(secret.exists())
            for i,item in enumerate(command[:-1]):
                if item=='--mount' and any('target='+target in command[i+1] for target in ['/checkpoint','/input','/assets','/adapter','/run/secrets']):
                    self.assertTrue(command[i+1].endswith(',readonly'))


if __name__=='__main__':unittest.main()
