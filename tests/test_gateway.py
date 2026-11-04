"""Local three-role integration: synthetic FFW inputs → HTTP model → FFW chunk executor."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
import uuid

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
from bg2_bridge.core import GROUPS,JointFeedback,Startup,all_names,load_home_yaml,BridgeError
from bg2_bridge.observations import Observations,decode_view
from bg2_bridge.gateway import Gateway,GatewayTransport,StatusCache
from bg2_bridge.chunks import ChunkExecutor
from bg2_bridge.rpc import InferenceClient
from inference_server import PolicyService,make_server
from test_network_inference import FakePolicy,TOKEN,observation

ROBOT=1_800_000_000_000_000_000
MONO=10_000_000_000
HOME=ROOT/'third_party/ai_worker/ffw_bringup/config/ffw_bg2_rev4_follower/ffw_bg2_follower_initial_positions.yaml'


def profile():
    value=json.loads((ROOT/'profiles/robotis_bg2_rev4.reference.json').read_text())
    for key,spec in value['cameras'].items():
        spec['raw_dimensions_verified']=True
        spec['live_to_training_rotation_deg']=0  # Explicit synthetic-fixture orientation.
        spec['raw_wh']=[672,376] if key=='head' else [240,424]
    return value


def image(key):
    wh=(672,376) if key=='head' else (240,424)
    color={'head':(255,0,0),'wrist':(0,0,255),'wrist_left':(0,255,0)}[key]
    stream=BytesIO();Image.new('RGB',wh,color).save(stream,format='PNG');return stream.getvalue()


def sample(p,source=ROBOT,received=MONO):
    names=all_names(p)
    return JointFeedback.from_arrays(names,[0.]*19,[0.]*19,source,received,p)


def ready(p):
    start=ROBOT-1_000_000_000
    value=Startup(p,load_home_yaml(HOME,p),start)
    for group in GROUPS:value.homing_status(group,4,start)
    for i in range(11):
        now=ROBOT-500_000_000+i*50_000_000
        value.update(sample(p,now,now),now)
    return value


def status(epoch,robot_now=ROBOT,mode='preview',phase='ready',remaining=0):
    return {'schema':'cosmos_bg2_status_v1','epoch':epoch,'robot_now_ns':robot_now,'mode':mode,'phase':phase,'remaining_ns':remaining}


class ObservationTests(unittest.TestCase):
    def test_three_views_match_rgb_layout_and_head_letterboxing(self):
        p=profile();obs=Observations(p);obs.add_joints(sample(p))
        for key in p['cameras']:obs.add_camera(key,ROBOT,MONO,image(key))
        result=obs.snapshot('Pick up the car.',ROBOT+10_000_000,receipt_now_ns=MONO+10_000_000)
        self.assertEqual(int(result['source_time_ns']),ROBOT)
        np.testing.assert_array_equal(result['head'][0,0],[0,0,0])
        np.testing.assert_array_equal(result['head'][20,20],[255,0,0])
        np.testing.assert_array_equal(result['wrist'][100,100],[0,0,255])
        np.testing.assert_array_equal(result['wrist_left'][100,100],[0,255,0])
        self.assertEqual(result['head'].shape,(384,512,3))
        self.assertEqual(result['wrist'].shape,(256,256,3))

    def test_stale_receipt_and_camera_skew_are_rejected(self):
        for camera_stamp,now in [(ROBOT-26_000_000,MONO),(ROBOT,MONO+101_000_000)]:
            p=profile();obs=Observations(p);obs.add_joints(sample(p))
            for key in p['cameras']:obs.add_camera(key,camera_stamp,MONO,image(key))
            with self.subTest(camera_stamp=camera_stamp),self.assertRaises(BridgeError):
                obs.snapshot('instruction',ROBOT,receipt_now_ns=now)

    def test_unknown_rotation_dimensions_or_wrong_crop_do_not_get_guessed(self):
        p=profile()
        for updates in [{'live_to_training_rotation_deg':None},{'raw_wh':[640,480]},{'crop_xyxy':[0,0,10,10]}]:
            spec={**p['cameras']['head'],**updates}
            with self.subTest(updates=updates),self.assertRaises(BridgeError):decode_view(image('head'),'head',spec)

    def test_explicit_rotation_is_applied_before_wrist_crop(self):
        stream=BytesIO();Image.new('RGB',(424,240),(25,50,75)).save(stream,format='PNG')
        spec={**profile()['cameras']['wrist'],'raw_wh':[424,240],'live_to_training_rotation_deg':270}
        result=decode_view(stream.getvalue(),'wrist',spec)
        self.assertEqual(result.shape,(256,256,3));np.testing.assert_array_equal(result[0,0],[25,50,75])


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.p=profile();self.startup=ready(self.p);self.robot=ChunkExecutor(self.startup)
        self.observations=Observations(self.p);self.observations.add_joints(sample(self.p))
        for key in self.p['cameras']:self.observations.add_camera(key,ROBOT,MONO,image(key))
        self.cache=StatusCache();self.cache.update(status(self.startup.epoch),MONO)
        self.service=PolicyService(FakePolicy,{},observation(),1)
        self.server=make_server(('127.0.0.1',0),self.service,TOKEN)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True);self.thread.start()
        self.client=InferenceClient(f'http://127.0.0.1:{self.server.server_address[1]}',TOKEN)
        self.sent=[]
        def forward(result):
            self.sent.append(result)
            self.robot.accept(result,sample(self.p,ROBOT+20_000_000,ROBOT+20_000_000),ROBOT+20_000_000)
        self.gateway=Gateway(self.observations,self.cache,self.client,forward,'Pick up the garbage truck.',latency_budget_ns=100_000_000)
    def tearDown(self):
        if self.service.policy.block is not None:self.service.policy.block.set()
        self.gateway.close();self.server.shutdown();self.server.server_close();self.thread.join(2);self.service.close()
    def finish(self,now=MONO+20_000_000):
        deadline=time.monotonic()+3
        event=None
        while self.gateway.future is not None and time.monotonic()<deadline:
            event=self.gateway.step(now);time.sleep(.005)
        self.assertIsNone(self.gateway.future)
        return event

    def test_raw_observation_to_http_to_ffw_executor_runs_without_robot_connection(self):
        self.gateway.step(MONO);self.finish()
        self.assertEqual(len(self.sent),1)
        self.assertEqual(self.sent[0]['source_time_ns'],ROBOT)
        self.assertEqual(self.sent[0]['epoch'],self.startup.epoch)
        self.assertEqual(self.robot.valid_until_ns,ROBOT+800_000_000)
        self.assertEqual(self.startup.phase,'running')
        self.assertEqual(len(self.robot.last_commands),4)
        self.assertEqual(self.service.calls,1)

    def test_stale_status_and_preview_execute_mismatch_do_not_start_requests(self):
        self.gateway.step(MONO+151_000_000)
        self.assertEqual(self.gateway.requests,0)
        self.cache.update(status(self.startup.epoch,mode='execute'),MONO+200_000_000)
        self.gateway.step(MONO+200_000_000)
        self.assertEqual(self.gateway.requests,0)

    def test_epoch_change_discards_inflight_result(self):
        self.service.policy.block=threading.Event();self.gateway.step(MONO)
        self.cache.update(status(uuid.uuid4().hex,ROBOT+10_000_000),MONO+10_000_000)
        self.service.policy.block.set();self.finish()
        self.assertEqual(self.sent,[])
        self.assertEqual(self.gateway.failed,1)

    def test_status_disappearing_discards_result(self):
        self.gateway.step(MONO);self.finish(MONO+200_000_000)
        self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)

    def test_prediction_that_arrives_near_expiry_is_not_forwarded(self):
        self.service.policy.block=threading.Event();self.gateway.step(MONO)
        self.cache.update(status(self.startup.epoch,ROBOT+750_000_000),MONO+10_000_000)
        self.service.policy.block.set();self.finish()
        self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)

    def test_connection_refused_forwards_nothing_and_recovers_with_a_new_observation(self):
        import socket
        original=self.gateway.client
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1',0))  # Reserve a local port without a listening service.
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{unavailable.getsockname()[1]}',TOKEN,timeout=.1)
            self.gateway.step(MONO);self.finish()
        self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)
        self.assertFalse(self.gateway.planner.in_flight)
        self.gateway.client=original
        self.gateway.send_chunk=self.sent.append
        later=130_000_000
        self.observations.add_joints(sample(self.p,ROBOT+later,MONO+later))
        for key in self.p['cameras']:self.observations.add_camera(key,ROBOT+later,MONO+later,image(key))
        self.cache.update(status(self.startup.epoch,ROBOT+later),MONO+later)
        self.gateway.step(MONO+later);self.finish(MONO+later+10_000_000)
        self.assertEqual(self.gateway.forwarded,1)
        self.assertEqual(self.sent[0]['source_time_ns'],ROBOT+later)
        self.assertEqual(self.gateway.requests,2)

    def test_dropped_tcp_connection_does_not_forward_or_leave_request_inflight(self):
        import socket
        import socketserver
        class Drop(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.recv(4096)
                self.request.shutdown(socket.SHUT_RDWR)
        server=socketserver.TCPServer(('127.0.0.1',0),Drop)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{server.server_address[1]}',TOKEN,timeout=.2)
            self.gateway.step(MONO);self.finish()
            self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)
            self.assertFalse(self.gateway.planner.in_flight)
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_unresponsive_http_service_times_out_without_forwarding(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        release=threading.Event();entered=threading.Event()
        class Hang(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                entered.set();release.wait(2)
            def log_message(self,*_):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Hang)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{server.server_address[1]}',TOKEN,timeout=.1)
            self.gateway.step(MONO);self.finish()
            self.assertTrue(entered.is_set())
            self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)
            self.assertFalse(self.gateway.planner.in_flight)
        finally:
            release.set();server.shutdown();server.server_close();thread.join(2)

    def test_no_request_queue_or_repeated_observation(self):
        self.service.policy.block=threading.Event();self.gateway.step(MONO)
        for _ in range(10):self.gateway.step(MONO+1)
        self.assertEqual(self.gateway.requests,1)
        self.service.policy.block.set();self.finish()
        for _ in range(10):self.gateway.step(MONO+21_000_000)
        self.assertEqual(self.gateway.requests,1)


class GatewayTransportTests(unittest.TestCase):
    def test_gateway_publishes_only_string_envelopes_and_inspect_creates_no_publishers(self):
        for inspect in [True,False]:
            created=[]
            class Endpoint:
                def __init__(self,**kwargs):created.append(kwargs)
                def close(self):pass
                def publish(self,**kwargs):pass
            sdk=NS(ROS2Subscriber=Endpoint,ROS2Publisher=lambda **kwargs:Endpoint(publisher=True,**kwargs))
            transport=GatewayTransport(profile(),Observations(profile()),StatusCache(),inspect=inspect,sdk=sdk)
            pubs=[x for x in created if x.get('publisher')]
            self.assertEqual(len(pubs),0 if inspect else 2)
            self.assertTrue(all(x['msg_type']=='std_msgs/msg/String' for x in pubs))
            self.assertTrue(all('/leader/' not in x['topic'] for x in pubs))
            transport.close()

    def test_dropped_invalid_packet_does_not_poison_later_valid_feedback(self):
        created=[]
        class Endpoint:
            def __init__(self,**kwargs):created.append(kwargs)
            def close(self):pass
        obs=Observations(profile())
        transport=GatewayTransport(profile(),obs,StatusCache(),inspect=True,sdk=NS(ROS2Subscriber=Endpoint))
        callback=next(x['callback'] for x in created if x['msg_type']=='sensor_msgs/msg/JointState')
        callback(NS(name=[],position=[],velocity=[],header=NS(stamp=NS(sec=1,nanosec=0))))
        self.assertEqual(transport.errors,1)
        callback(NS(name=all_names(profile()),position=[0.]*19,velocity=[0.]*19,
                    header=NS(stamp=NS(sec=1800000000,nanosec=0))))
        self.assertEqual(obs.latest_source_ns(),ROBOT)
        self.assertIsNone(obs.error)
        transport.close()


class GatewayBundleTests(unittest.TestCase):
    def test_bundle_contains_only_required_code_and_passes_offline_validation(self):
        import hashlib
        import subprocess
        import tempfile
        from package_gateway import build
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'gateway'
            manifest=build(out)
            self.assertLess(sum(v['bytes'] for v in manifest['files'].values()),600_000)
            for name,expected in manifest['files'].items():
                path=Path(name)
                self.assertNotIn('.git',path.parts)
                self.assertNotIn('__pycache__',path.parts)
                self.assertNotEqual(path.name,'inference.token')
                self.assertEqual(hashlib.sha256((out/path).read_bytes()).hexdigest(),expected['sha256'])
            self.assertTrue((out/'third_party/zenoh_ros2_sdk/LICENSE').is_file())
            self.assertEqual(len(list((out/'messages').rglob('*.msg'))),5)
            self.assertFalse(json.loads((out/'config/profile.reference.json').read_text())['execution_enabled'])
            result=subprocess.run([sys.executable,'-B',str(out/'scripts/gateway.py'),'inspect','--profile',str(out/'config/profile.reference.json'),
                '--messages-dir',str(out/'messages'),'--output',str(out/'unused'),'--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)['robot_connection_opened'])
            self.assertFalse((out/'unused').exists())
            with self.assertRaises(ValueError):build(out)

    def test_gateway_compose_needs_no_devices_and_mounts_token_read_only(self):
        import yaml
        value=yaml.safe_load((ROOT/'deploy/gateway/compose.yaml').read_text())['services']['gateway']
        self.assertNotIn('devices',value)
        self.assertNotIn('privileged',value)
        self.assertNotIn('ports',value)
        self.assertTrue(value['read_only'])
        self.assertIn('${COSMOS_MODE:-preview}',value['command'])
        for mount in value['volumes']:
            self.assertFalse(mount['bind']['create_host_path'])
            if mount['target'].startswith('/config/'):
                self.assertTrue(mount['read_only'])


@unittest.skipUnless(importlib.util.find_spec('zenoh') and importlib.util.find_spec('rosbags'),'Optional isolated SDK environment not active')
class RealSdkTests(unittest.TestCase):
    def test_pinned_sdk_delivers_using_only_bundled_message_definitions(self):
        import os
        from unittest import mock
        sys.path.insert(0,str(ROOT/'third_party/zenoh_ros2_sdk'))
        import zenoh_ros2_sdk as sdk
        import zenoh_ros2_sdk._cache as cache
        message=threading.Event();seen=[]
        config='mode="peer";connect/endpoints=[];listen/endpoints=["tcp/127.0.0.1:0"];scouting/multicast/enabled=false;transport/shared_memory/enabled=false'
        with mock.patch.dict(os.environ,{'ZENOH_CONFIG_OVERRIDE':config}),mock.patch.object(cache,'clone_to_cache',side_effect=AssertionError('Runtime message download forbidden')):
            sdk.get_registry(messages_dir=str(ROOT/'deploy/gateway/messages'))
            def receive(msg):seen.append(msg.data);message.set()
            subscriber=sdk.ROS2Subscriber(topic='/cosmos_local_test/string',msg_type='std_msgs/msg/String',callback=receive)
            publisher=sdk.ROS2Publisher(topic='/cosmos_local_test/string',msg_type='std_msgs/msg/String')
            try:
                publisher.publish(data='local-only')
                self.assertTrue(message.wait(2));self.assertEqual(seen,['local-only'])
            finally:
                publisher.close();subscriber.close();sdk.ZenohSession.get_instance().close();sdk.ZenohSession._instance=None


if __name__=='__main__':unittest.main()
