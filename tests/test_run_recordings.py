"""Local recording/control isolation, transfer, aggregation and cancellation tests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from bg2_bridge.artifact_protocol import atomic_json
from bg2_bridge.rpc import InferenceClient
from bg2_bridge.rollouts import RunRollouts,render_chunk
from inference_server import PolicyService,make_server
from test_network_inference import FakePolicy,observation
from test_wam_service import FakeWAM


@contextmanager
def server(factory,root=None):
    service=PolicyService(factory,{},observation(),1,root)
    http=make_server(('127.0.0.1',0),service)
    thread=threading.Thread(target=http.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:yield service,f'http://127.0.0.1:{http.server_address[1]}'
    finally:
        http.shutdown();http.server_close();thread.join(2);service.close()


def wait_state(client,reference,wanted,seconds=10):
    end=time.monotonic()+seconds
    while True:
        result=client.recording_status(reference['status_path'])
        if result['state']==wanted:return result
        if time.monotonic()>end:raise AssertionError(result)
        time.sleep(.01)


class RunRecordingTests(unittest.TestCase):
    def test_blocked_encoder_does_not_block_first_or_next_prediction(self):
        release=threading.Event();entered=threading.Event()
        with tempfile.TemporaryDirectory() as root:
            with server(FakeWAM,root) as (service,url):
                def render(folder):entered.set();release.wait(30)
                service.recordings.renderer=render
                client=InferenceClient(url,timeout=10,run_id='execute-ambulance')
                try:
                    first=client.predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)
                    self.assertTrue(entered.wait(2));self.assertFalse(release.is_set())
                    second=client.predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)
                    self.assertEqual(client.health()['prediction_calls'],2)
                    self.assertEqual(client.recording_status(first['artifacts']['status_path'])['state'],'rendering')
                    self.assertEqual(client.recording_status(second['artifacts']['status_path'])['state'],'pending')
                    self.assertEqual(np.load(Path(root)/first['artifacts']['directory']/'decoded_rollout.npy').shape,(17,640,512,3))
                finally:
                    release.set();service.recordings.jobs.join()

    def test_ctrl_c_can_interrupt_an_inflight_120_second_http_wait(self):
        with server(FakePolicy) as (service,url):
            block=threading.Event();service.policy.block=block
            client=InferenceClient(url,timeout=120)
            with ThreadPoolExecutor(1) as pool:
                future=pool.submit(client.predict,observation(),uuid.uuid4().hex,uuid.uuid4().hex)
                deadline=time.monotonic()+2
                while service.policy.calls<2 and time.monotonic()<deadline:time.sleep(.005)
                client.cancel()
                try:
                    with self.assertRaises(Exception):future.result(timeout=1)
                finally:block.set()

    def test_downloads_group_by_run_and_remux_two_forecasts_in_order(self):
        import av
        class Colors(FakeWAM):
            def predict(self,obs,**kwargs):
                result=super().predict(obs,**kwargs)
                result['imagined_video'][:]=[220,0,0] if obs['joint_position'][0]==0 else [0,0,220]
                return result
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);run=root/'execute-cars';run.mkdir()
            with server(Colors,root/'pro') as (service,url):
                client=InferenceClient(url,timeout=5,run_id=run.name)
                collector=RunRollouts(run,client.recording_client())
                for index in range(2):
                    obs=observation(index+100);obs['joint_position'][0]=index
                    epoch,ident=uuid.uuid4().hex,uuid.uuid4().hex
                    collector.register(epoch,ident,int(obs['source_time_ns']))
                    response=client.predict(obs,epoch,ident)
                    collector.prediction(response)
                collector.close(wait=20)
                data=json.loads((run/'imagined/manifest.json').read_text())
                self.assertEqual([e['state'] for e in data['chunks']],['complete','complete'])
                for name,wh in [('rollout.mp4',(512,640)),('head.mp4',(512,384)),('wrist_left.mp4',(256,256)),('wrist_right.mp4',(256,256))]:
                    with av.open(str(run/'imagined'/name)) as video:
                        frames=list(video.decode(video=0));stream=video.streams.video[0]
                        self.assertEqual(len(frames),32);self.assertEqual((stream.width,stream.height),wh)
                        self.assertEqual(float(stream.average_rate),20.)
                        self.assertGreater(int(frames[0].to_ndarray(format='rgb24')[20,20,0]),200)
                        self.assertGreater(int(frames[16].to_ndarray(format='rgb24')[20,20,2]),200)
                timeline=json.loads((run/'imagined/timeline.json').read_text())
                self.assertEqual(timeline['frame_count'],32);self.assertFalse(timeline['partial'])
                self.assertEqual(timeline['chunks'][1]['first_frame'],16)

    def test_pending_collection_can_resume_after_gateway_exits(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);run=root/'debug-resume';run.mkdir()
            with server(FakeWAM,root/'pro') as (service,url):
                release=threading.Event();original=service.recordings.renderer
                def delayed(folder):release.wait(5);original(folder)
                service.recordings.renderer=delayed
                client=InferenceClient(url,timeout=5,run_id=run.name)
                collector=RunRollouts(run,client.recording_client())
                epoch,ident=uuid.uuid4().hex,uuid.uuid4().hex
                collector.register(epoch,ident,100)
                result=client.predict(observation(100),epoch,ident);collector.prediction(result)
                collector.close(wait=0)
                self.assertEqual(collector.pending(),1)
                release.set()
                resumed=RunRollouts(run,client.recording_client(),resume=True)
                resumed.close(wait=20)
                self.assertEqual(resumed.pending(),0)
                self.assertTrue((run/'imagined/rollout.mp4').exists())

    def test_recording_api_rejects_traversal_without_exposing_other_files(self):
        with tempfile.TemporaryDirectory() as root:
            with server(FakeWAM,root) as (service,url):
                client=InferenceClient(url,timeout=5,run_id='bad/../run')
                with self.assertRaises(ValueError):client.predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)
                self.assertEqual(service.calls,0)

    def test_collects_a_legacy_timed_out_request_by_id_without_running_model(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);run=root/'execute-old';old=root/'pro'
            epoch,ident=uuid.uuid4().hex,uuid.uuid4().hex
            inputs=run/'predictions'/ident;inputs.mkdir(parents=True)
            atomic_json(inputs/'observation.json',{'source_time_ns':123})
            archived=old/'requests'/epoch/ident;archived.mkdir(parents=True)
            np.save(archived/'decoded_rollout.npy',np.zeros((17,640,512,3),np.uint8))
            render_chunk(archived)
            atomic_json(archived/'prediction.json',{'mode':'wam','epoch':epoch,'request_id':ident,'source_time_ns':123})
            atomic_json(archived/'model_output.json',{'mode':'wam'})
            atomic_json(archived/'rollout.json',{'fps':20})
            with server(FakeWAM,old) as (service,url):
                collector=RunRollouts(run,InferenceClient(url,5),resume=True)
                collector.close(wait=10)
                self.assertEqual(collector.pending(),0)
                self.assertEqual(service.calls,0)
                saved=json.loads((run/'imagined/manifest.json').read_text())['chunks'][0]
                self.assertEqual(saved['epoch'],epoch)
                self.assertFalse(saved['prediction_received'])
                self.assertTrue((run/'imagined/rollout.mp4').is_file())

    def test_recovers_legacy_rgb_even_if_old_mp4_export_never_completed(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            with server(FakeWAM,root) as (service,url):
                epoch,ident=uuid.uuid4().hex,uuid.uuid4().hex
                folder=root/'requests'/epoch/ident;(folder/'inputs').mkdir(parents=True)
                np.savez(folder/'inputs/observation.npz',**observation(456))
                np.save(folder/'decoded_rollout.npy',np.zeros((17,640,512,3),np.uint8))
                atomic_json(folder/'prediction.json',{'mode':'wam','epoch':epoch,'request_id':ident,'source_time_ns':456})
                client=InferenceClient(url,5)
                from bg2_bridge.artifact_protocol import artifact_path
                saved=client.recording_status(artifact_path('execute-recovered',epoch,ident))
                self.assertEqual(saved['state'],'complete')
                self.assertIn('decoded_rollout.npy',saved['files'])
                self.assertFalse(any(name.endswith('.mp4') for name in saved['files']))
                self.assertEqual(json.loads((folder/'rollout.json').read_text())['forecast_time_ns'][-1],800000456)
                self.assertEqual(service.calls,0)
