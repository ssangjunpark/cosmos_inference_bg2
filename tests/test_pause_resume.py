"""No-robot checks for current-pose startup and pause → new prompt → fresh observation."""
import json
import importlib.util
from io import BytesIO
from PIL import Image
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import math
import time
import unittest
import uuid
from unittest import mock

from bg2_bridge.core import BridgeError, SafetyStop, Startup, GROUPS, load_home_yaml
from bg2_bridge.chunks import ChunkExecutor
from bg2_bridge.gateway import Gateway, StatusCache
from gateway import finish_control
from test_ffw_companion import profile, feedback, ready, chunk, HOME, BASE
from test_gateway import profile as camera_profile, sample, status, image, MONO, ROBOT
from bg2_bridge.observations import Observations
from test_network_inference import FakePolicy, observation
from inference_server import PolicyService


class CurrentPoseTests(unittest.TestCase):
    def startup(self):
        p=profile();return p,Startup(p,load_home_yaml(HOME,p),BASE,require_homing=False)

    def test_captures_actual_non_home_pose_without_any_homing_status(self):
        p,start=self.startup();pose={name:0. for name in feedback(p,BASE).positions}
        pose.update(arm_r_joint1=.4,arm_l_joint2=.5,lift_joint=-.18,head_joint1=.2)
        for i in range(11):
            now=BASE+i*50_000_000;start.update(feedback(p,now,pose),now)
            if i<10:self.assertIsNone(start.hold)
        self.assertEqual(start.phase,'ready');self.assertEqual(start.hold,pose)

    def test_motion_and_feedback_gaps_reset_settling(self):
        p,start=self.startup();start.update(feedback(p,BASE),BASE)
        start.update(feedback(p,BASE+400_000_000,velocities={'arm_r_joint1':2.01}),BASE+400_000_000)
        self.assertIsNone(start.hold)
        for i in range(10):
            now=BASE+450_000_000+i*50_000_000;start.update(feedback(p,now),now)
        self.assertIsNone(start.hold)
        now=BASE+950_000_000;start.update(feedback(p,now),now);self.assertEqual(start.phase,'ready')

    def test_measured_gripper_jitter_does_not_block_current_pose_startup(self):
        p,start=self.startup()
        for i in range(11):
            now=BASE+i*50_000_000;pose=dict(feedback(p,now).positions)
            pose['gripper_l_joint1']=.0015339807878858025*(i%2)
            frame=feedback(p,now,pose,{'gripper_l_joint1':.0716283126,'gripper_r_joint1':.0955044168})
            start.update(frame,now)
        self.assertEqual(start.phase,'ready');self.assertEqual(start.hold,pose)

    def test_range_overshoot_for_every_joint_is_preserved_without_clipping(self):
        for boundary,offset in [('upper',.001),('lower',-.001)]:
            p,start=self.startup()
            pose={name:p['reference_limits'][name][boundary]+offset for name in feedback(p,BASE).positions}
            for i in range(11):
                now=BASE+i*50_000_000;frame=feedback(p,now,pose);start.update(frame,now)
            self.assertEqual(start.phase,'ready');self.assertEqual(start.hold,pose)
            engine=ChunkExecutor(start);commands=engine.tick(frame,now)
            for group,spec in p['commands'].items():
                self.assertEqual(commands[group]['message']['points'][0]['positions'],[pose[n] for n in spec['joint_names']])
            data=chunk(start,now,cycle=engine.cycle)
            self.assertEqual(engine.accept(data,frame,now)['arm_right']['message']['points'][0]['positions'],data['positions'][0])

    def test_in_progress_homing_blocks_capture_even_if_goal_predates_startup(self):
        p,start=self.startup();start.homing_status('arm_right',2,BASE-1)
        for i in range(12):
            now=BASE+i*50_000_000;start.update(feedback(p,now),now)
        self.assertIsNone(start.hold)
        start.homing_status('arm_right',4,BASE-1)
        for i in range(11):
            now=BASE+600_000_000+i*50_000_000;start.update(feedback(p,now),now)
        self.assertEqual(start.phase,'ready')

    def test_failed_active_homing_is_not_ignored(self):
        _,start=self.startup();start.homing_status('arm_right',2,BASE-1)
        with self.assertRaises(BridgeError):start.homing_status('arm_right',6,BASE-1)
        self.assertEqual(start.phase,'fault')


class PauseTests(unittest.TestCase):
    def engine(self):
        start,frame=ready();engine=ChunkExecutor(start);engine.tick(frame,frame.source_ns)
        return start,frame,engine

    def test_pause_interrupts_chunk_at_measured_pose_and_invalidates_old_replies(self):
        start,frame,engine=self.engine();old=chunk(start,frame.source_ns)
        engine.accept(old,frame,frame.source_ns)
        now=frame.source_ns+200_000_000;pose=dict(frame.positions);pose.update(engine.expected_right(now));pose["arm_l_joint1"] = .04
        measured=feedback(start.profile,now,pose);submitted=[];previous=start.epoch
        commands=engine.pause(measured,now,submit=submitted.append)
        self.assertEqual(start.phase,'paused');self.assertNotEqual(start.epoch,previous)
        self.assertFalse(engine.status(now)['needs_chunk']);self.assertEqual(engine.execution_end_ns,0)
        for group,spec in start.profile['commands'].items():
            self.assertEqual(commands[group]['message']['points'][0]['positions'],[pose[n] for n in spec['joint_names']])
        self.assertIsNone(engine.tick(measured,now))
        with self.assertRaises(BridgeError):engine.accept(old,measured,now)
        self.assertEqual(len(submitted),1)

    def test_resume_preserves_pose_and_requires_post_resume_observation(self):
        start,frame,engine=self.engine();engine.pause(frame,frame.source_ns)
        paused_epoch=start.epoch;now=frame.source_ns+50_000_000;measured=feedback(start.profile,now,frame.positions)
        engine.resume(measured,now)
        self.assertEqual(start.phase,'holding');self.assertNotEqual(start.epoch,paused_epoch)
        self.assertEqual(engine.ready_since_ns,now);self.assertTrue(engine.status(now)['needs_chunk'])
        self.assertEqual(start.hold,frame.positions)
        for source in [now-1,now]:
            data=chunk(start,source,index=source,cycle=engine.cycle)
            if source<now:
                with self.assertRaises(BridgeError):engine.accept(data,measured,now)
            else:engine.accept(data,measured,now)
        self.assertEqual(start.phase,'running')
        self.assertEqual(len(engine.last_commands['arm_right']['message']['points']),16)

    def test_fault_cannot_resume_and_paused_monitoring_still_applies(self):
        start,frame,engine=self.engine();engine.pause(frame,frame.source_ns)
        bad=feedback(start.profile,frame.source_ns+1,velocities={'arm_r_joint1':2.01})
        with self.assertRaises(SafetyStop):engine.resume(bad,bad.source_ns)
        engine.stop(frame,frame.source_ns,fault=True)
        with self.assertRaises(BridgeError):engine.resume(frame,frame.source_ns)

    def test_failed_pause_publication_does_not_advertise_paused(self):
        start,frame,engine=self.engine();old=start.epoch
        with self.assertRaises(OSError):engine.pause(frame,frame.source_ns,submit=mock.Mock(side_effect=OSError('lost connection')))
        self.assertEqual(start.phase,'ready');self.assertEqual(start.epoch,old)

    def test_external_homing_during_pause_invalidates_hold(self):
        start,frame,engine=self.engine();engine.pause(frame,frame.source_ns)
        with self.assertRaises(BridgeError):start.homing_status('head',2,frame.source_ns+1)
        self.assertEqual(start.phase,'fault');self.assertIsNone(start.hold)


class GatewayResumeTests(unittest.TestCase):
    def test_new_gateway_resumes_once_then_uses_new_prompt_and_post_resume_sensors(self):
        p=camera_profile();obs=Observations(p);cache=StatusCache();controls=[];sent=[]
        def add(source,received):
            obs.add_joints(sample(p,source,received))
            for key in p['cameras']:obs.add_camera(key,source,received,image(key))
        add(ROBOT,MONO)
        paused=uuid.uuid4().hex
        cache.update({**status(paused,mode='execute',phase='paused',ready_since=0),'controllers_ready':True},MONO)
        service=PolicyService(FakePolicy,{},observation(),1)
        observed_prompts=[]
        original_predict=service.predict
        def record_prompt(obs,epoch,ident):
            observed_prompts.append(str(obs['instruction']))
            return original_predict(obs,epoch,ident)
        service.predict=record_prompt
        try:
            gateway=Gateway(obs,cache,service,sent.append,'New task instruction','execute',send_control=lambda epoch,cmd:controls.append((epoch,cmd)))
            self.assertEqual(gateway.step(MONO)['event'],'resume_requested')
            gateway.step(MONO+1);self.assertEqual(controls,[(paused,'resume')]);self.assertEqual(service.calls,0)
            # A dropped resume envelope is retried with the same paused epoch.
            gateway.step(MONO+110_000_000);self.assertEqual(len(controls),2)
            later=120_000_000;resumed=uuid.uuid4().hex
            cache.update(status(resumed,ROBOT+later,mode='execute',phase='holding',cycle=2,ready_since=ROBOT+later),MONO+later)
            gateway.step(MONO+later);self.assertEqual(service.calls,0)
            add(ROBOT+later,MONO+later)
            result=gateway.step(MONO+later);self.assertEqual(result['event'],'forwarded')
            self.assertEqual(sent[0]['epoch'],resumed);self.assertEqual(sent[0]['source_time_ns'],ROBOT+later)
            self.assertEqual(len(sent[0]['positions']),16)
            self.assertEqual(observed_prompts,['New task instruction'])
        finally:service.close()

    def test_waiting_reports_actual_ffw_initialization_phase(self):
        cache=StatusCache();value=status(uuid.uuid4().hex,mode='execute',phase='settling',ready_since=0)
        value['reason']='Waiting for settled current joint feedback';cache.update(value,MONO)
        engine=Gateway(mock.Mock(),cache,mock.Mock(),mock.Mock(),'task','execute')
        event=engine.step(MONO)
        self.assertEqual(event['ffw_phase'],'settling');self.assertEqual(event['reason'],value['reason'])

    def test_existing_gateway_cannot_undo_a_pause_or_resume_a_fault(self):
        cache=StatusCache();epoch=uuid.uuid4().hex
        cache.update({**status(epoch,mode='execute',phase='paused',ready_since=0),'controllers_ready':True},MONO)
        control=mock.Mock();engine=Gateway(mock.Mock(),cache,mock.Mock(),mock.Mock(),'task','execute',send_control=control)
        engine.requests=1;engine.step(MONO);control.assert_not_called()
        engine.requests=0
        cache.update({**status(epoch,mode='execute',phase='fault',ready_since=0),'controllers_ready':True},MONO)
        engine.step(MONO);control.assert_not_called()

    def test_ctrl_c_cleanup_requires_an_observed_pause_acknowledgement(self):
        cache=StatusCache();old=uuid.uuid4().hex;now=time.monotonic_ns()
        cache.update(status(old,mode='execute',phase='holding'),now)
        transport=mock.Mock()
        def pause(epoch,command):
            self.assertEqual((epoch,command),(old,'pause'))
            cache.update(status(uuid.uuid4().hex,mode='execute',phase='paused',ready_since=0),time.monotonic_ns())
        transport.control.side_effect=pause
        result=finish_control(transport,cache,'pause')
        self.assertEqual(result['event'],'pause_confirmed');transport.control.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('av'),'PyAV required for finalization validation')
    def test_cli_ctrl_c_pauses_and_finalizes_all_camera_files(self):
        import gateway as cli
        import av
        p=camera_profile();p.update(execution_enabled=True,status='operator_enabled_for_execution')
        handlers={};transport=mock.Mock();transport.diagnostics.return_value={};cache=None
        def connect(profile,observations,status,**kwargs):
            nonlocal cache
            cache=status
            cache.update(globals()['status'](uuid.uuid4().hex,mode='execute',phase='holding'),time.monotonic_ns())
            for key in profile['cameras']:
                packet=BytesIO()
                with Image.open(BytesIO(image(key))) as frame:frame.save(packet,format='JPEG')
                kwargs['recorder'].camera(key,ROBOT,time.monotonic_ns(),packet.getvalue())
            return transport
        def control(epoch,command):
            self.assertEqual(command,'pause')
            cache.update(status(uuid.uuid4().hex,mode='execute',phase='paused',ready_since=0),time.monotonic_ns())
        transport.control.side_effect=control
        class Engine:
            def __init__(self,*args,**kwargs):pass
            def close(self):pass
            def step(self,*args):
                handlers[cli.signal.SIGINT]()
                return {'event':'interrupted'}
        client=mock.Mock();client.health.return_value={'records_outputs':True}
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'profile.json';config.write_text(json.dumps(p));output=Path(temp)/'run'
            with mock.patch.object(cli,'GatewayTransport',side_effect=connect),mock.patch.object(cli,'Gateway',Engine),mock.patch.object(cli,'InferenceClient',return_value=client),mock.patch.object(cli.signal,'signal',side_effect=lambda sig,fn:handlers.update({sig:fn})):
                cli.main(['execute','--profile',str(config),'--instruction','new task','--output',str(output)])
            self.assertEqual(json.loads((output/'recording_summary.json').read_text())['frames'],{'head':1,'wrist_left':1,'wrist':1})
            for key in p['cameras']:
                with av.open(str(output/'cameras'/(key+'.mkv'))) as video:self.assertEqual(len(list(video.decode(video=0))),1)
            self.assertIn('pause_confirmed',(output/'events.jsonl').read_text())
            transport.control.assert_called_once();transport.send_chunk.assert_not_called()
            transport.close.assert_called_once()

    def test_no_ack_is_reported_without_claiming_the_robot_paused(self):
        cache=StatusCache();cache.update(status(uuid.uuid4().hex,mode='execute',phase='holding'),time.monotonic_ns())
        with mock.patch('gateway.time.monotonic',side_effect=[0.,2.]):
            result=finish_control(mock.Mock(),cache,'pause')
        self.assertEqual(result['event'],'control_not_confirmed')


if __name__=='__main__':unittest.main()
