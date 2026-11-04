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
from bg2_bridge.chunks import ChunkExecutor, EXECUTION_CONTRACT, STATUS_SCHEMA
from bg2_bridge.rpc import InferenceClient
from inference_server import PolicyService,make_server
from test_network_inference import FakePolicy,observation

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


def status(epoch,robot_now=ROBOT,mode='preview',phase='ready',remaining=0,cycle=0,ready_since=ROBOT,completed=None):
    return {'schema':STATUS_SCHEMA,'execution_contract':EXECUTION_CONTRACT,'epoch':epoch,'robot_now_ns':robot_now,'mode':mode,'phase':phase,'remaining_ns':remaining,'cycle':cycle,'ready_since_ns':ready_since,'needs_chunk':phase in {'ready','holding'},'last_completed_request_id':completed}


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

    def test_sync_snapshot_requires_all_camera_and_joint_stamps_after_hold(self):
        p=profile();obs=Observations(p);obs.add_joints(sample(p,ROBOT+10_000_000,MONO+10_000_000))
        for key in p['cameras']:obs.add_camera(key,ROBOT,MONO,image(key))
        with self.assertRaises(BridgeError):
            obs.snapshot('instruction',ROBOT+10_000_000,receipt_now_ns=MONO+10_000_000,not_before_ns=ROBOT+1)
        for key in p['cameras']:obs.add_camera(key,ROBOT+10_000_000,MONO+10_000_000,image(key))
        result=obs.snapshot('instruction',ROBOT+10_000_000,receipt_now_ns=MONO+10_000_000,not_before_ns=ROBOT+1)
        self.assertEqual(int(result['source_time_ns']),ROBOT+10_000_000)

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
        self.robot.tick(sample(self.p,ROBOT,ROBOT),ROBOT)
        self.observations=Observations(self.p);self.add_observation(ROBOT,MONO)
        self.cache=StatusCache();self.cache.update(status(self.startup.epoch),MONO)
        self.service=PolicyService(FakePolicy,{},observation(),1)
        self.server=make_server(('127.0.0.1',0),self.service)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True);self.thread.start()
        self.client=InferenceClient(f'http://127.0.0.1:{self.server.server_address[1]}')
        self.sent=[];self.robot_now=ROBOT+20_000_000
        def forward(result):
            self.sent.append(result)
            self.robot.accept(result,sample(self.p,self.robot_now,self.robot_now),self.robot_now)
        self.gateway=Gateway(self.observations,self.cache,self.client,forward,'Pick up the garbage truck.')
    def tearDown(self):
        if self.service.policy.block is not None:self.service.policy.block.set()
        self.gateway.close();self.server.shutdown();self.server.server_close();self.thread.join(2);self.service.close()
    def add_observation(self,source,received):
        self.observations.add_joints(sample(self.p,source,received))
        for key in self.p['cameras']:self.observations.add_camera(key,source,received,image(key))

    def test_raw_observation_to_http_to_full_ffw_chunk(self):
        event=self.gateway.step(MONO)
        self.assertEqual(event['event'],'forwarded');self.assertEqual(len(self.sent),1)
        self.assertEqual(self.robot.execution_end_ns,self.robot_now+800_000_000)
        self.assertEqual(self.sent[0]['source_time_ns'],ROBOT)
        self.assertEqual(self.startup.phase,'running');self.assertEqual(self.service.calls,1)
        self.assertEqual(len(self.robot.last_commands['arm_right']['message']['points']),16)

    def test_no_requests_during_execution_or_before_completion_acknowledgement(self):
        self.gateway.step(MONO)
        self.add_observation(ROBOT+30_000_000,MONO+30_000_000)
        # Even an old ready status must not cause a second request after forwarding.
        self.gateway.step(MONO+30_000_000);self.assertEqual(self.gateway.requests,1)
        self.cache.update(status(self.startup.epoch,ROBOT+40_000_000,phase='running'),MONO+40_000_000)
        self.gateway.step(MONO+40_000_000);self.assertEqual(self.gateway.requests,1)

    def test_next_request_only_after_final_hold_with_a_new_observation(self):
        self.gateway.step(MONO)
        end=self.robot.execution_end_ns
        self.robot.tick(sample(self.p,end,end),end)
        state=self.robot.status(end);state['mode']='preview'
        elapsed=end-ROBOT
        self.cache.update(state,MONO+elapsed)
        self.gateway.step(MONO+elapsed)  # No post-hold cameras/joints yet.
        self.assertEqual(self.gateway.requests,1)
        self.add_observation(end,MONO+elapsed);self.robot_now=end+20_000_000
        event=self.gateway.step(MONO+elapsed)
        self.assertEqual(event['event'],'forwarded');self.assertEqual(self.gateway.requests,2)
        self.assertEqual(self.sent[-1]['cycle'],1);self.assertEqual(self.sent[-1]['source_time_ns'],end)

    def test_slow_inference_keeps_all_points_instead_of_expiring_them(self):
        # Keep the FFW hold heartbeat fresh throughout a >0.8-second model request.
        block=threading.Event();self.service.policy.block=block
        result=[];start=time.monotonic_ns()
        self.observations=Observations(self.p);self.gateway.observations=self.observations
        self.add_observation(ROBOT,start);self.cache.update(status(self.startup.epoch),start)
        t=threading.Thread(target=lambda:result.append(self.gateway.step()));t.start()
        while time.monotonic_ns()-start<850_000_000:
            elapsed=time.monotonic_ns()-start
            self.cache.update(status(self.startup.epoch,ROBOT+elapsed),start+elapsed)
            time.sleep(.01)
        elapsed=time.monotonic_ns()-start;self.robot_now=ROBOT+elapsed
        block.set();t.join(3)
        self.assertFalse(t.is_alive());self.assertEqual(result[0]['event'],'forwarded',result)
        self.assertEqual(len(self.robot.last_commands['arm_right']['message']['points']),16)
        self.assertEqual(self.robot.execution_end_ns,self.robot_now+800_000_000)

    def test_stale_status_mode_mismatch_and_running_never_start_requests(self):
        self.gateway.step(MONO+151_000_000)
        for updates in [{'mode':'execute'},{'phase':'running'},{'phase':'fault'}]:
            self.cache.update(status(self.startup.epoch,**updates),MONO)
            self.gateway.step(MONO)
        self.assertEqual(self.gateway.requests,0)

    def test_old_async_status_is_rejected(self):
        old=status(self.startup.epoch);old['schema']='cosmos_bg2_status_v1'
        with self.assertRaises(BridgeError):self.cache.update(old,MONO)

    def test_session_change_or_user_stop_during_inference_never_forwards(self):
        original=self.gateway.client
        class Changed:
            def predict(inner,obs,epoch,ident):
                result=original.predict(obs,epoch,ident)
                self.cache.update(status(uuid.uuid4().hex),MONO)
                return result
        self.gateway.client=Changed();event=self.gateway.step(MONO)
        self.assertEqual(event['event'],'discarded');self.assertEqual(self.sent,[])

    def test_explicit_close_during_inference_never_forwards(self):
        original=self.gateway.client
        class Closed:
            def predict(inner,obs,epoch,ident):
                result=original.predict(obs,epoch,ident);self.gateway.close();return result
        self.gateway.client=Closed();self.gateway.step(MONO)
        self.assertEqual(self.sent,[])

    def test_connection_refusal_recovers_with_a_new_held_observation(self):
        import socket
        original=self.gateway.client
        with socket.socket() as unavailable:
            unavailable.bind(('127.0.0.1',0))
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{unavailable.getsockname()[1]}',timeout=.1)
            self.gateway.step(MONO)
        self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1)
        self.assertFalse(self.gateway.in_flight)
        self.gateway.client=original;self.gateway.send_chunk=self.sent.append
        later=130_000_000;self.add_observation(ROBOT+later,MONO+later)
        self.cache.update(status(self.startup.epoch,ROBOT+later),MONO+later)
        self.gateway.step(MONO+later)
        self.assertEqual(self.gateway.forwarded,1);self.assertEqual(self.gateway.requests,2)

    def test_dropped_tcp_connection_and_timeout_forward_nothing(self):
        import socket,socketserver
        class Drop(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.recv(4096);self.request.shutdown(socket.SHUT_RDWR)
        server=socketserver.TCPServer(('127.0.0.1',0),Drop)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{server.server_address[1]}',timeout=.1)
            self.gateway.step(MONO)
            self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1);self.assertFalse(self.gateway.in_flight)
        finally:server.shutdown();server.server_close();thread.join(2)

    def test_http_timeout_does_not_forward_or_leave_request_running(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        release=threading.Event()
        class Hang(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']));release.wait(2)
            def log_message(self,*_):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Hang)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            self.gateway.client=InferenceClient(f'http://127.0.0.1:{server.server_address[1]}',timeout=.1)
            self.gateway.step(MONO)
            self.assertEqual(self.sent,[]);self.assertEqual(self.gateway.failed,1);self.assertFalse(self.gateway.in_flight)
        finally:release.set();server.shutdown();server.server_close();thread.join(2)


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

    def test_gateway_compose_needs_no_devices_or_password_file(self):
        import yaml
        value=yaml.safe_load((ROOT/'deploy/gateway/compose.yaml').read_text())['services']['gateway']
        self.assertNotIn('devices',value)
        self.assertNotIn('privileged',value)
        self.assertNotIn('ports',value)
        self.assertTrue(value['read_only'])
        self.assertIn('${COSMOS_MODE:-preview}',value['command'])
        self.assertNotIn('--token-file',value['command'])
        self.assertFalse(any(m['target']=='/config/inference.token' for m in value['volumes']))
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
