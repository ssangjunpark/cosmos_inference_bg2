"""5090 synchronous observation → inference → full-chunk gateway."""
from io import BytesIO
from pathlib import Path
import json
import os
import sys
import threading
import time
import uuid

from PIL import Image
from .core import JointFeedback, require, stamp_ns
from .chunks import EXECUTION_CONTRACT, STATUS_SCHEMA
from .observations import Observations
from .rpc import identifier
from .recording import RecordingError


class StatusCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.value, self.received_ns = None, 0

    def update(self, value, received_ns):
        require(isinstance(value, dict) and value.get('schema') == STATUS_SCHEMA and
                value.get('execution_contract') == EXECUTION_CONTRACT, 'Synchronous FFW status required')
        identifier(value.get('epoch'))
        require(value.get('phase') in {'initializing','settling','ready','running','holding','fault','stopped'}, 'Invalid FFW phase')
        require(value.get('mode') in {'preview','execute'}, 'Invalid FFW mode')
        require(type(value.get('robot_now_ns')) is int and value['robot_now_ns'] > 0, 'Invalid robot status clock')
        require(type(value.get('cycle')) is int and value['cycle'] >= 0, 'Invalid execution cycle')
        require(type(value.get('ready_since_ns')) is int and value['ready_since_ns'] >= 0, 'Invalid hold timestamp')
        require(type(value.get('needs_chunk')) is bool, 'Invalid request readiness')
        if value['needs_chunk']:
            require(value['phase'] in {'ready','holding'} and 0 < value['ready_since_ns'] <= value['robot_now_ns'], 'Invalid hold readiness')
        with self.lock:
            if self.value and value['epoch'] == self.value['epoch']:
                require(value['robot_now_ns'] >= self.value['robot_now_ns'] and value['cycle'] >= self.value['cycle'], 'Out-of-order FFW status')
            self.value, self.received_ns = dict(value), received_ns

    def get(self, now_ns):
        with self.lock:
            if self.value is None or not 0 <= now_ns - self.received_ns <= 150_000_000:
                return None
            return dict(self.value), self.received_ns


class Gateway:
    def __init__(self, observations, status, client, send_chunk, instruction, mode='preview', recorder=None):
        self.observations, self.status, self.client = observations, status, client
        self.send_chunk, self.instruction, self.mode = send_chunk, instruction, mode
        self.recorder = recorder
        self.epoch, self.awaiting, self.last_source_ns = None, None, 0
        self.forwarded, self.requests, self.failed, self.next_retry_ns = 0, 0, 0, 0
        self.closed, self.in_flight = False, False

    def eligible(self, cached):
        return cached is not None and cached[0]['mode'] == self.mode and cached[0]['needs_chunk'] and cached[0]['phase'] in {'ready','holding'}

    def step(self, now_ns=None):
        if self.recorder is not None:self.recorder.check()
        entered_ns = time.monotonic_ns()
        now_ns = entered_ns if now_ns is None else now_ns
        cached = self.status.get(now_ns)
        if self.closed or self.in_flight or not self.eligible(cached):
            return {'event':'waiting','reason':'Waiting for FFW to finish all points and hold'}
        current, received = cached
        if current['epoch'] != self.epoch:
            self.epoch, self.awaiting, self.last_source_ns = current['epoch'], None, 0
        if self.awaiting is not None:
            cycle, request_id = self.awaiting
            if current['cycle'] <= cycle:
                return {'event':'waiting','reason':'Waiting for FFW chunk completion acknowledgement'}
            if current.get('last_completed_request_id') != request_id:
                return {'event':'waiting','reason':'FFW completion does not match the submitted chunk'}
            self.awaiting = None
        if now_ns < self.next_retry_ns:
            return None
        try:
            if self.observations.latest_source_ns() <= self.last_source_ns:
                return None
            robot_now = current['robot_now_ns'] + (now_ns - received)
            observation = self.observations.snapshot(self.instruction, robot_now, receipt_now_ns=now_ns,
                                                     not_before_ns=current['ready_since_ns'], include_raw=self.recorder is not None)
            if self.recorder is not None:observation,raw_frames=observation
            source = int(observation['source_time_ns'])
            if source <= self.last_source_ns:
                return None
        except (ValueError, OSError) as error:
            return {'event':'waiting','reason':str(error)}
        self.last_source_ns = source
        request_id, epoch, cycle = uuid.uuid4().hex, current['epoch'], current['cycle']
        self.requests += 1
        self.in_flight = True
        started = time.monotonic_ns()
        try:
            if self.recorder is not None:self.recorder.inputs(request_id,observation,raw_frames)
            # Blocking inference: FFW is holding, and no next request is prefetched.
            result = self.client.predict(observation, epoch, request_id)
            if self.recorder is not None:self.recorder.prediction(result)
            elapsed = max(1, time.monotonic_ns() - started)
            latest = self.status.get(now_ns + time.monotonic_ns() - entered_ns)
            require(not self.closed and self.eligible(latest) and latest[0]['epoch'] == epoch and
                    latest[0]['cycle'] == cycle, 'FFW session/readiness changed while inference was running')
            result = {**result, 'execution_contract':EXECUTION_CONTRACT, 'cycle':cycle}
            self.send_chunk(result)
            self.awaiting = cycle, request_id
            self.forwarded += 1
            return {'event':'forwarded','request_id':request_id,'epoch':epoch,'cycle':cycle,
                    'source_time_ns':source,'roundtrip_seconds':elapsed/1e9,'timings':result.get('timings',{})}
        except RecordingError:
            self.closed = True
            raise
        except Exception as error:
            self.failed += 1
            self.next_retry_ns = now_ns + max(1, time.monotonic_ns() - entered_ns) + 100_000_000
            return {'event':'discarded','reason':str(error)}
        finally:
            self.in_flight = False

    def close(self):
        self.closed = True


class GatewayTransport:
    def __init__(self, profile, observations, status, *, inspect=False, sdk=None, messages_dir=None, recorder=None):
        self.profile,self.observations,self.status=profile,observations,status
        self.subscribers,self.chunk_publisher,self.control_publisher=[],None,None
        self.recorder=recorder
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
        source,receipt,payload=stamp_ns(msg.header.stamp),time.monotonic_ns(),bytes(msg.data)
        self.observations.add_camera(key,source,receipt,payload)
        if self.recorder is not None:self.recorder.camera(key,source,receipt,payload)

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


class DebugGateway:
    """Live policy prediction and recording only; no FFW status, controller or publishers."""
    def __init__(self,observations,client,recorder,instruction):
        self.observations,self.client,self.recorder,self.instruction=observations,client,recorder,instruction
        self.epoch=uuid.uuid4().hex
        self.closed,self.last_source_ns,self.requests,self.predictions=False,0,0,0

    def close(self):self.closed=True

    def step(self,now_ns=None):
        if self.closed:return None
        self.recorder.check()
        now_ns=time.monotonic_ns() if now_ns is None else now_ns
        with self.observations.lock:
            if not self.observations.joints:return {'event':'waiting','reason':'Waiting for joint feedback'}
            anchor=self.observations.joints[-1].source_ns
        if anchor<=self.last_source_ns:return None
        try:
            observation,raw=self.observations.snapshot(self.instruction,anchor,receipt_now_ns=now_ns,include_raw=True)
        except (ValueError,OSError) as error:return {'event':'waiting','reason':str(error)}
        request_id=uuid.uuid4().hex;self.last_source_ns=int(observation['source_time_ns'])
        self.recorder.inputs(request_id,observation,raw);self.requests+=1
        result=self.client.predict(observation,self.epoch,request_id)
        self.recorder.prediction(result);self.predictions+=1
        return {'event':'prediction_saved','request_id':request_id,'epoch':self.epoch,
                'source_time_ns':self.last_source_ns,'physical_publishing':False,'artifacts':result.get('artifacts')}
