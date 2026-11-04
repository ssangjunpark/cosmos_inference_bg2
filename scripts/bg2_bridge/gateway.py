"""5090 gateway: observations and asynchronous RPC only; no joint command publishers."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import json
import os
import sys
import threading
import time
import uuid

from PIL import Image

from .core import BridgeError, JointFeedback, require, stamp_ns
from .chunks import RequestPlanner
from .observations import Observations
from .rpc import identifier


class StatusCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.value, self.received_ns = None, 0

    def update(self, value, received_ns):
        require(isinstance(value, dict) and value.get('schema') == 'cosmos_bg2_status_v1', 'Unknown FFW status')
        identifier(value.get('epoch'))
        require(value.get('phase') in {'initializing','settling','ready','running','holding','fault','stopped'}, 'Invalid FFW phase')
        require(value.get('mode') in {'preview','execute'}, 'Invalid FFW mode')
        require(type(value.get('robot_now_ns')) is int and value['robot_now_ns'] > 0, 'Invalid robot status clock')
        require(type(value.get('remaining_ns')) is int and 0 <= value['remaining_ns'] <= 800_000_000, 'Invalid remaining horizon')
        with self.lock:
            if self.value and value['epoch'] == self.value['epoch']:
                require(value['robot_now_ns'] >= self.value['robot_now_ns'], 'Out-of-order FFW status')
            self.value, self.received_ns = dict(value), received_ns

    def get(self, now_ns):
        with self.lock:
            if self.value is None or not 0 <= now_ns - self.received_ns <= 150_000_000:
                return None
            return dict(self.value), self.received_ns


class Gateway:
    def __init__(self, observations, status, client, send_chunk, instruction, mode='preview', latency_budget_ns=200_000_000):
        self.observations, self.status, self.client = observations, status, client
        self.send_chunk, self.instruction, self.mode = send_chunk, instruction, mode
        self.planner = RequestPlanner(latency_budget_ns)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cosmos-request')
        self.future, self.context, self.epoch = None, None, None
        self.forwarded, self.requests, self.failed, self.next_retry_ns = 0, 0, 0, 0

    def step(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        cached = self.status.get(now_ns)
        eligible = cached is not None and cached[0]['mode'] == self.mode and cached[0]['phase'] in {'ready','running','holding'}
        event = None
        if eligible:
            current, received = cached
            if current['epoch'] != self.epoch:
                self.epoch = current['epoch']
                self.planner = RequestPlanner(self.planner.latency_budget_ns, in_flight=self.future is not None)
        if self.future is not None and self.future.done():
            context, future = self.context, self.future
            self.future, self.context = None, None
            elapsed = max(1, now_ns - context['started_ns'])
            self.planner.finished(elapsed)
            try:
                result = future.result()
                require(eligible and current['epoch'] == context['epoch'], 'Discarded result from an old/unavailable FFW session')
                robot_now = current['robot_now_ns'] + (now_ns - received)
                require(result['source_time_ns'] + 800_000_000 - robot_now >= 100_000_000, 'Discarded late prediction')
                self.send_chunk(result)
                self.forwarded += 1
                event = {'event':'forwarded','request_id':result['request_id'],'epoch':result['epoch'],
                         'source_time_ns':result['source_time_ns'],'roundtrip_seconds':elapsed/1e9,'timings':result.get('timings',{})}
            except Exception as error:
                self.failed += 1
                self.next_retry_ns = now_ns + 100_000_000
                event = {'event':'discarded','reason':str(error),'roundtrip_seconds':elapsed/1e9}
        if self.future is None and eligible and now_ns >= self.next_retry_ns:
            try:
                source_hint = self.observations.latest_source_ns()
                if not self.planner.should_request(current, received, now_ns, source_hint):
                    return event
                robot_now = current['robot_now_ns'] + (now_ns - received)
                observation = self.observations.snapshot(self.instruction, robot_now, receipt_now_ns=now_ns)
                source = int(observation['source_time_ns'])
                if source - self.planner.last_source_ns >= 50_000_000 and self.planner.should_request(current,received,now_ns,source):
                    request_id = uuid.uuid4().hex
                    self.planner.started(source)
                    self.context = {'epoch':current['epoch'],'started_ns':now_ns,'request_id':request_id}
                    self.future = self.pool.submit(self.client.predict,observation,current['epoch'],request_id)
                    self.requests += 1
                    event = event or {'event':'requested','request_id':request_id,'epoch':current['epoch'],'source_time_ns':source}
            except (ValueError, OSError) as error:
                event = event or {'event':'waiting','reason':str(error)}
        if not eligible:
            event = event or {'event':'waiting','reason':'Waiting for fresh FFW status in the matching preview/execute mode'}
        return event

    def close(self):
        if self.future is not None:self.future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)


class GatewayTransport:
    def __init__(self, profile, observations, status, *, inspect=False, sdk=None, messages_dir=None):
        self.profile,self.observations,self.status=profile,observations,status
        self.subscribers,self.chunk_publisher,self.control_publisher=[],None,None
        self.errors,self.last_error=0,None
        self.node_name='cosmos_gateway_'+uuid.uuid4().hex[:12]
        middleware=profile['middleware']
        self.options={'domain_id':middleware['domain_id'],'router_ip':middleware['configured_router_host'],
                      'router_port':middleware['configured_router_port']}
        if sdk is None:
            root=Path(__file__).resolve().parents[2]
            sdk_path=root/'third_party/zenoh_ros2_sdk'
            if sdk_path.is_dir():sys.path.insert(0,str(sdk_path))
            os.environ['ZENOH_CONFIG_OVERRIDE']=('mode="client";connect/endpoints='+
                json.dumps([f"tcp/{self.options['router_ip']}:{self.options['router_port']}"])+
                ';scouting/multicast/enabled=false;transport/shared_memory/enabled=false')
            import zenoh_ros2_sdk as sdk
            require(sdk.__version__=='0.1.8','Use the pinned ROBOTIS SDK')
        self.sdk=sdk
        if messages_dir is not None:sdk.get_registry(messages_dir=str(messages_dir))
        topics=profile.get('bridge_topics',{})
        try:
            spec=profile['joint_feedback']
            self._subscribe(spec['topic'],spec['msg_type'],self._joints)
            for key,camera in profile['cameras'].items():
                self._subscribe(camera['topic'],camera['msg_type'],lambda msg,key=key:self._camera(key,msg))
            if not inspect:
                self._subscribe(topics.get('status','/cosmos/bg2/status'),'std_msgs/msg/String',self._status)
                # Only JSON chunk/control envelopes. Never publish JointTrajectory or
                # create controller clients on the 5090.
                self.chunk_publisher=sdk.ROS2Publisher(topic=topics.get('chunk','/cosmos/bg2/chunk'),
                    msg_type='std_msgs/msg/String',node_name=self.node_name,**self.options)
                self.control_publisher=sdk.ROS2Publisher(topic=topics.get('control','/cosmos/bg2/control'),
                    msg_type='std_msgs/msg/String',node_name=self.node_name,**self.options)
        except Exception:
            self.close();raise

    def _subscribe(self,topic,msg_type,callback):
        def checked(msg):
            try:callback(msg)
            except Exception as error:
                self.errors+=1;self.last_error=str(error)
        self.subscribers.append(self.sdk.ROS2Subscriber(topic=topic,msg_type=msg_type,callback=checked,
                                                       node_name=self.node_name,**self.options))

    def _joints(self,msg):
        self.observations.add_joints(JointFeedback.from_arrays(msg.name,msg.position,msg.velocity,
                                    stamp_ns(msg.header.stamp),time.monotonic_ns(),self.profile))

    def _camera(self,key,msg):
        self.observations.add_camera(key,stamp_ns(msg.header.stamp),time.monotonic_ns(),bytes(msg.data))

    def _status(self,msg):
        require(len(msg.data.encode())<=128*1024,'Oversized FFW status')
        self.status.update(json.loads(msg.data),time.monotonic_ns())

    def send_chunk(self,result):
        require(self.chunk_publisher is not None,'Inspect mode cannot forward chunks')
        self.chunk_publisher.publish(data=json.dumps(result,allow_nan=False))

    def stop(self,epoch):
        if self.control_publisher is not None:
            self.control_publisher.publish(data=json.dumps({'schema':'cosmos_bg2_control_v1','epoch':epoch,'command':'stop'}))

    def diagnostics(self,now_ns):
        with self.observations.lock:
            result={'joint_stamp_ns':self.observations.joints[-1].source_ns if self.observations.joints else None,
                    'camera':{},'dropped_messages':self.errors,'last_error':self.last_error}
            frames={key:items[-1] if items else None for key,items in self.observations.cameras.items()}
        for key,frame in frames.items():
            if frame is None:result['camera'][key]=None;continue
            with Image.open(BytesIO(frame[2])) as image:wh=list(image.size)
            result['camera'][key]={'source_time_ns':frame[0],'receipt_age_ms':(now_ns-frame[1])/1e6,'raw_wh':wh,
                                  'configured_rotation_deg':self.profile['cameras'][key].get('live_to_training_rotation_deg')}
        return result

    def close(self):
        for sub in self.subscribers:sub.close()
        self.subscribers.clear()
        for pub in (self.chunk_publisher,self.control_publisher):
            if pub is not None:pub.close()
        self.chunk_publisher=self.control_publisher=None
