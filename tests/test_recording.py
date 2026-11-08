"""Debug isolation, exact prediction evidence, and compressed camera recordings."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest import mock
import uuid

import numpy as np
from PIL import Image

from bg2_bridge.core import BridgeError
from bg2_bridge.gateway import DebugGateway, Gateway, GatewayTransport, StatusCache
from bg2_bridge.observations import Observations
from bg2_bridge.rpc import InferenceClient
from bg2_bridge.recording import CameraVideo, RecordingError, RunRecorder
from inference_server import PolicyService
from test_gateway import MONO, ROBOT, image, profile, sample, status
from test_network_inference import FakePolicy, observation
import gateway as cli


class EvidencePolicy(FakePolicy):
    def predict(self, obs, **kwargs):
        return {**super().predict(obs, **kwargs), 'vision':[np.zeros((1,48,1,40,32),np.float32)],
                'normalized':np.ones((17,64),np.float32), 'prompt':str(obs['instruction'])}


class RecordingTests(unittest.TestCase):
    def test_server_records_exact_actions_inputs_and_context_latent_without_rollout(self):
        with tempfile.TemporaryDirectory() as temp:
            service=PolicyService(EvidencePolicy,{'seed':0},observation(),1,temp)
            try:
                obs=observation();obs['joint_position'][0]=.123
                epoch,ident=uuid.uuid4().hex,uuid.uuid4().hex
                result=service.predict(obs,epoch,ident)
                self.assertTrue(service.health()['records_outputs'])
                folder=Path(temp)/result['artifacts']['directory']
                meta=json.loads((folder/'model_output.json').read_text())
                self.assertEqual(meta['mode'],'policy');self.assertFalse(meta['imagined_future_rollout'])
                self.assertEqual(meta['visuals'][0]['kind'],'conditioned_context_latent')
                self.assertEqual(meta['visuals'][0]['decoded_frames'],0)
                self.assertEqual(meta['prompt'],str(obs['instruction']))
                self.assertEqual(json.loads((folder/'prediction.json').read_text())['positions'],result['positions'])
                with np.load(folder/'inputs/observation.npz',allow_pickle=False) as saved:
                    for key in obs:np.testing.assert_array_equal(saved[key],obs[key])
                with Image.open(folder/'inputs/input_composite.png') as composite:
                    self.assertEqual(composite.size,(512,640))
                self.assertEqual(np.load(folder/'vision_000_latent.npy',allow_pickle=False).shape,(1,48,1,40,32))
                self.assertFalse(list(folder.glob('*decoded*')))
            finally:service.close()

    def test_no_overwrite_of_a_recorded_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder=RunRecorder(temp,'debug')
            try:
                recorder.inputs('first',observation(),{})
                with self.assertRaises(RecordingError):recorder.inputs('first',observation(),{})
                with self.assertRaises(RecordingError):recorder.check()
            finally:recorder.close()

    @unittest.skipUnless(importlib.util.find_spec('av'),'PyAV required for media validation')
    def test_all_three_camera_streams_remux_original_packets_and_keep_exact_stamps(self):
        import av
        with tempfile.TemporaryDirectory() as temp:
            recorder=RunRecorder(temp,'debug');expected={}
            try:
                for key in ('head','wrist_left','wrist'):
                    packets=[]
                    for i,offset in enumerate((0,33_333_333,83_900_001)):
                        buf=BytesIO();Image.new('RGB',(32,24),(i*60,25,50)).save(buf,format='JPEG')
                        payload=buf.getvalue();packets.append(payload)
                        recorder.camera(key,ROBOT+offset,MONO+offset,payload)
                    expected[key]=packets
            finally:recorder.close()
            for key,packets in expected.items():
                with av.open(str(Path(temp)/'cameras'/(key+'.mkv'))) as video:
                    actual=[bytes(p) for p in video.demux(video.streams.video[0]) if p.size]
                self.assertEqual(actual,packets)
                with av.open(str(Path(temp)/'cameras'/(key+'.mkv'))) as video:
                    self.assertEqual(len(list(video.decode(video=0))),3)
                stamps=[json.loads(x) for x in (Path(temp)/'cameras'/(key+'.timestamps.jsonl')).read_text().splitlines()]
                self.assertEqual([x['source_time_ns'] for x in stamps],[ROBOT+x for x in (0,33_333_333,83_900_001)])
            self.assertEqual(json.loads((Path(temp)/'recording_summary.json').read_text())['frames'],dict.fromkeys(expected,3))

    @unittest.skipUnless(importlib.util.find_spec('av'),'PyAV required for media validation')
    def test_media_failure_latches_error_and_closes_files(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder=RunRecorder(temp,'debug')
            with self.assertRaises(RecordingError):recorder.camera('head',ROBOT,MONO,b'invalid image')
            with self.assertRaises(RecordingError):recorder.check()
            recorder.close();recorder.close()
            self.assertIsNotNone(json.loads((Path(temp)/'recording_summary.json').read_text())['error'])


class DebugTests(unittest.TestCase):
    def observations(self, receipt=MONO):
        p=profile();obs=Observations(p);obs.add_joints(sample(p,ROBOT,receipt))
        for key in p['cameras']:obs.add_camera(key,ROBOT,receipt,image(key))
        return p,obs

    def test_debug_needs_no_homing_or_ffw_status_and_saves_exact_selected_frames(self):
        p,obs=self.observations()
        with tempfile.TemporaryDirectory() as temp:
            service=PolicyService(FakePolicy,{},observation(),1)
            recorder=RunRecorder(temp,'debug')
            try:
                engine=DebugGateway(obs,service,recorder,'Pick up the police car and place it on the top-right picture')
                event=engine.step(MONO)
                self.assertEqual(event['event'],'prediction_saved');self.assertFalse(event['physical_publishing'])
                self.assertEqual(engine.predictions,1);self.assertIsNone(engine.step(MONO))
                folder=Path(temp)/'predictions'/event['request_id']
                self.assertEqual(json.loads((folder/'prediction.json').read_text())['positions'],[[0.]*8]*16)
                for key in p['cameras']:self.assertEqual((folder/('raw_'+key+'.png')).read_bytes(),image(key))
                engine.close();self.assertIsNone(engine.step(MONO))
            finally:recorder.close();service.close()

    def test_debug_refuses_stale_live_inputs_without_making_a_prediction(self):
        p,obs=self.observations()
        with tempfile.TemporaryDirectory() as temp:
            client=mock.Mock();recorder=RunRecorder(temp,'debug')
            try:
                event=DebugGateway(obs,client,recorder,'instruction').step(MONO+101_000_000)
                self.assertEqual(event['event'],'waiting');client.predict.assert_not_called()
            finally:recorder.close()

    def test_recording_failure_prevents_normal_gateway_forwarding(self):
        p,obs=self.observations();cache=StatusCache();cache.update(status(uuid.uuid4().hex),MONO)
        recorder=mock.Mock();recorder.prediction.side_effect=RecordingError('disk full')
        client=mock.Mock();client.predict.return_value={'request_id':uuid.uuid4().hex}
        send=mock.Mock();engine=Gateway(obs,cache,client,send,'instruction',recorder=recorder)
        with self.assertRaises(RecordingError):engine.step(MONO)
        send.assert_not_called();self.assertTrue(engine.closed)

    def test_transport_has_only_four_sensor_subscribers_in_debug(self):
        p,obs=self.observations();created=[]
        class Endpoint:
            def __init__(self,**kw):created.append(kw)
            def close(self):pass
        sdk=NS(ROS2Subscriber=Endpoint,ROS2Publisher=mock.Mock(side_effect=AssertionError('No publishers in debug')))
        transport=GatewayTransport(p,obs,StatusCache(),inspect=True,sdk=sdk)
        try:
            self.assertEqual(len(created),4)
            self.assertEqual({x['msg_type'] for x in created},{'sensor_msgs/msg/JointState','sensor_msgs/msg/CompressedImage'})
            with self.assertRaises(BridgeError):transport.send_chunk({})
            transport.stop(uuid.uuid4().hex);sdk.ROS2Publisher.assert_not_called()
        finally:transport.close()

    def test_cli_execute_debug_ignores_execution_enable_but_cannot_send_or_stop(self):
        p,obs=self.observations(time.monotonic_ns());self.assertFalse(p['execution_enabled'])
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'profile.json';config.write_text(json.dumps(p))
            output=Path(temp)/'run';transport=mock.Mock()
            def connect(*args,**kwargs):
                self.assertTrue(kwargs['inspect']);return transport
            class Client:
                def __init__(self,*args,**kwargs):pass
                def health(self):return {'records_outputs':True}
                def predict(self,observed,epoch,ident):
                    return {'epoch':epoch,'request_id':ident,'positions':[[0.]*8]*16}
            with mock.patch.object(cli,'Observations',return_value=obs),mock.patch.object(cli,'InferenceClient',Client),mock.patch.object(cli,'GatewayTransport',side_effect=connect),mock.patch.object(cli.signal,'signal'):
                cli.main(['execute','--debug','--instruction','instruction','--profile',str(config),'--output',str(output),'--max-predictions','1'])
            transport.send_chunk.assert_not_called();transport.stop.assert_not_called();transport.close.assert_called_once()
            self.assertFalse(json.loads((output/'run.json').read_text())['physical_publishing'])
            self.assertEqual(len(list((output/'predictions').iterdir())),1)

    def test_early_http_rejection_survives_a_broken_upload_pipe(self):
        response=mock.Mock(status=503)
        response.read.return_value=b'{"ok":false,"error":"Model is busy"}'
        connection=mock.Mock()
        connection.request.side_effect=BrokenPipeError('server replied before upload ended')
        connection.getresponse.return_value=response
        with mock.patch('bg2_bridge.rpc.http.client.HTTPConnection',return_value=connection):
            with self.assertRaisesRegex(BridgeError,'Model is busy'):
                InferenceClient('http://127.0.0.1:8008').predict(observation(),uuid.uuid4().hex,uuid.uuid4().hex)
        connection.close.assert_called_once()

    def test_debug_disconnected_service_closes_media_before_exiting(self):
        p,obs=self.observations()
        with tempfile.TemporaryDirectory() as temp:
            recorder=RunRecorder(temp,'debug');client=mock.Mock()
            client.predict.side_effect=ConnectionResetError('connection dropped')
            try:
                with self.assertRaises(ConnectionResetError):DebugGateway(obs,client,recorder,'instruction').step(MONO)
            finally:recorder.close()
            self.assertTrue((Path(temp)/'recording_summary.json').is_file())


if __name__=='__main__':unittest.main()
