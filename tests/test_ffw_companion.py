"""Offline FFW chunk lifecycle and portable-bundle checks; never opens a robot connection."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bg2_bridge.core import BridgeError, GROUPS, JointFeedback, Startup, all_names, load_home_yaml, validate_profile, SafetyStop
from bg2_bridge.chunks import ChunkExecutor, EXECUTION_CONTRACT, SCHEMA
from package_ffw import build

HOME = ROOT / "third_party/ai_worker/ffw_bringup/config/ffw_bg2_rev4_follower/ffw_bg2_follower_initial_positions.yaml"
BASE = 10_000_000_000


def profile():
    return json.loads((ROOT / "profiles/robotis_bg2_rev4.reference.json").read_text())


def feedback(p, now, positions=None, velocities=None):
    names = all_names(p)
    positions = positions or {name: 0. for name in names}
    return JointFeedback.from_arrays(names, [positions[n] for n in names], [0.] * len(names) if velocities is None else [velocities.get(n, 0.) for n in names], now, now, p)


def ready(p=None, custom=None):
    p = p or profile()
    home = load_home_yaml(HOME, p)
    if custom:
        for data in home.values():
            for name, value in custom.items():
                if name in data["positions"]:
                    data["positions"][name] = value
    startup = Startup(p, home, BASE)
    for group in GROUPS:
        startup.homing_status(group, 4, BASE)
    pose = {name: value for data in home.values() for name, value in data["positions"].items()}
    for i in range(12):
        frame = feedback(p, BASE + i * 50_000_000, pose)
        startup.update(frame, frame.source_ns)
    return startup, frame


def chunk(startup, source, index=1, step=.001, cycle=0):
    right = startup.profile["policy"]["joint_names"]
    rows = [[startup.hold[n] for n in right] for _ in range(16)]
    for i, row in enumerate(rows):
        row[0] += step * (i + 1)
    return {"schema": SCHEMA, "execution_contract": EXECUTION_CONTRACT, "cycle": cycle, "epoch": startup.epoch, "request_id": f"{index:032x}",
            "source_time_ns": source, "joint_names": right, "units": ["rad"] * 8, "positions": rows}


class HomingTests(unittest.TestCase):
    def test_real_duplicate_root_yaml_preserves_all_groups(self):
        home = load_home_yaml(HOME, profile())
        self.assertEqual(set(home), set(GROUPS))
        self.assertEqual(sum(len(x["positions"]) for x in home.values()), 19)
        self.assertEqual(home["arm_right"]["status_topic"], "/arm_r_controller/follow_joint_trajectory/_action/status")

    def test_homing_success_alone_does_not_capture_before_settling(self):
        p = profile(); startup = Startup(p, load_home_yaml(HOME, p), BASE)
        for group in GROUPS:
            startup.homing_status(group, 4, BASE)
        self.assertIsNone(startup.hold)
        startup.update(feedback(p, BASE), BASE)
        self.assertEqual(startup.phase, "settling")
        self.assertIsNone(startup.hold)

    def test_custom_home_captured_pose_stays_fixed(self):
        startup, frame = ready(custom={"arm_l_joint2": .4, "lift_joint": -.18, "head_joint1": .1})
        captured = copy.deepcopy(startup.hold)
        changed = copy.deepcopy(frame.positions); changed["arm_l_joint2"] += .005
        startup.update(feedback(startup.profile, frame.source_ns + 50_000_000, changed), frame.source_ns + 50_000_000)
        self.assertEqual(startup.hold, captured)
        self.assertEqual(startup.hold["lift_joint"], -.18)

    def test_incomplete_or_old_homing_cannot_open_gate(self):
        p = profile(); startup = Startup(p, load_home_yaml(HOME, p), BASE)
        for group in GROUPS:
            startup.homing_status(group, 4, BASE - 1)
        self.assertEqual(startup.phase, "initializing")
        startup.homing_status("arm_right", 4, BASE)
        self.assertEqual(startup.phase, "initializing")

    def test_external_reinitialization_invalidates_session_and_old_hold(self):
        startup, _ = ready()
        epoch = startup.epoch
        with self.assertRaisesRegex(BridgeError, "New homing"):
            startup.homing_status("arm_right", 2, BASE + 1_000_000_000)
        self.assertEqual(startup.phase, "fault")
        self.assertIsNone(startup.hold)
        self.assertNotEqual(startup.epoch, epoch)

    def test_failed_homing_does_not_capture(self):
        p = profile(); startup = Startup(p, load_home_yaml(HOME, p), BASE)
        with self.assertRaisesRegex(BridgeError, "Homing failed"):
            startup.homing_status("arm_left", 6, BASE)
        self.assertIsNone(startup.hold)

    def test_feedback_gap_resets_settling(self):
        p = profile(); startup = Startup(p, load_home_yaml(HOME, p), BASE)
        for group in GROUPS: startup.homing_status(group, 4, BASE)
        startup.update(feedback(p, BASE), BASE)
        startup.update(feedback(p, BASE + 600_000_000), BASE + 600_000_000)
        self.assertIsNone(startup.hold)
        self.assertEqual(startup.phase, "settling")

    def test_operator_head_override_preserves_other_limits_and_captured_hold(self):
        from prepare_execution_profile import prepare
        from bg2_bridge.core import check_positions
        p = profile()
        for key, camera in p['cameras'].items():
            camera.update(raw_dimensions_verified=True, raw_wh=[672,376] if key=='head' else [424,240],
                          live_to_training_rotation_deg=0 if key=='head' else 90)
        overridden = prepare(p, snupi_head_override=True)
        validate_profile(overridden, live=True, execute=True)
        self.assertEqual(overridden['status'], 'operator_enabled_for_execution')
        self.assertEqual(overridden['reference_limits']['head_joint1']['upper'], .78)
        self.assertEqual(p['reference_limits']['head_joint1']['upper'], .6951)
        for name, limit in p['reference_limits'].items():
            if name != 'head_joint1':self.assertEqual(overridden['reference_limits'][name],limit)
        check_positions(['head_joint1'], [.78], overridden)
        with self.assertRaises(BridgeError):check_positions(['head_joint1'], [.7801], overridden)
        startup, frame = ready(p=overridden, custom={'head_joint1': .78})
        engine = ChunkExecutor(startup)
        commands = engine.tick(frame, frame.source_ns)
        self.assertEqual(commands['head']['message']['points'][0]['positions'][0], .78)

    def test_execution_profile_still_requires_verified_cameras_and_explicit_enable(self):
        from prepare_execution_profile import prepare
        with self.assertRaises(BridgeError):prepare(profile(), snupi_head_override=True)
        p = profile();p['status'] = 'operator_enabled_for_execution'
        with self.assertRaises(BridgeError):validate_profile(p, execute=True)

    def test_reference_profile_cannot_enable_execution(self):
        with self.assertRaises(BridgeError):
            validate_profile(profile(), execute=True)


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.startup, self.frame = ready()
        self.engine = ChunkExecutor(self.startup)
        self.source = self.frame.source_ns
        self.engine.tick(self.frame, self.source)

    def accept(self, data=None, now=None, submit=None):
        now = self.source + 20_000_000 if now is None else now
        return self.engine.accept(data or chunk(self.startup, self.source),
                                  feedback(self.startup.profile, now, self.frame.positions), now, submit=submit)

    def test_all_16_targets_unchanged_and_timed_from_execution_even_after_slow_inference(self):
        data = chunk(self.startup, self.source)
        now = self.source + 2_000_000_000
        msg = self.accept(data, now)["arm_right"]["message"]
        self.assertEqual(len(msg["points"]), 16)
        self.assertEqual([p["positions"] for p in msg["points"]], data["positions"])
        offsets = [p["time_from_start"]["sec"] * 10**9 + p["time_from_start"]["nanosec"] for p in msg["points"]]
        self.assertEqual(offsets, list(range(50_000_000, 800_000_001, 50_000_000)))
        self.assertEqual(msg["header"]["stamp"]["sec"] * 10**9 + msg["header"]["stamp"]["nanosec"], now)
        self.assertEqual(self.engine.execution_end_ns, now + 800_000_000)

    def test_active_chunk_cannot_be_replaced(self):
        previous = self.accept()
        with self.assertRaisesRegex(BridgeError, "finish"):
            self.accept(chunk(self.startup, self.source + 100_000_000, index=2), self.source + 120_000_000)
        self.assertIs(self.engine.last_commands, previous)

    def test_completion_holds_last_command_not_measured_pose_then_opens_next_cycle(self):
        data=chunk(self.startup,self.source); self.accept(data)
        end=self.engine.execution_end_ns
        self.assertIsNone(self.engine.tick(feedback(self.startup.profile,end-1,self.frame.positions),end-1))
        result=self.engine.tick(feedback(self.startup.profile,end,self.frame.positions),end)
        self.assertEqual(self.startup.phase,'holding'); self.assertEqual(self.engine.cycle,1)
        self.assertEqual(result['arm_right']['message']['points'][0]['positions'],data['positions'][-1])
        self.assertNotEqual(result['arm_right']['message']['points'][0]['positions'][0],self.frame.positions['arm_r_joint1'])
        for group in self.startup.profile['hold_groups']:
            names=self.startup.profile['commands'][group]['joint_names']
            self.assertEqual(result[group]['message']['points'][0]['positions'],[self.startup.hold[n] for n in names])
        self.assertEqual(self.engine.last_completed_request_id,data['request_id'])
        self.assertTrue(self.engine.status(end)['needs_chunk'])
        self.assertIsNone(self.engine.tick(feedback(self.startup.profile,end+1,self.frame.positions),end+1))

    def test_next_observation_must_follow_hold_and_match_cycle(self):
        self.accept(); end=self.engine.execution_end_ns
        self.engine.tick(feedback(self.startup.profile,end,self.frame.positions),end)
        for source,cycle in [(end-1,1),(end,0)]:
            with self.subTest(source=source,cycle=cycle),self.assertRaises(BridgeError):
                self.accept(chunk(self.startup,source,index=2,cycle=cycle),end)
        self.accept(chunk(self.startup,end,index=3,cycle=1),end)
        self.assertEqual(self.startup.phase,'running')

    def test_failed_hold_publication_does_not_announce_completion(self):
        self.accept(); end=self.engine.execution_end_ns
        def fail(_): raise RuntimeError('hold publish failed')
        with self.assertRaises(RuntimeError):
            self.engine.tick(feedback(self.startup.profile,end,self.frame.positions),end,submit=fail)
        self.assertFalse(self.engine.status(end)['needs_chunk'])
        self.assertEqual(self.engine.cycle,0)
        self.assertIsNone(self.engine.last_completed_request_id)

    def test_commanded_velocity_limit_is_absolute_2_radians_not_quarter_of_urdf(self):
        for delta,allowed in [(.09,True),(.1,True),(.10001,False)]:
            startup,frame=ready(); engine=ChunkExecutor(startup);engine.tick(frame,frame.source_ns)
            data=chunk(startup,frame.source_ns)
            for row in data['positions']: row[0]=frame.positions['arm_r_joint1']+delta
            with self.subTest(delta=delta):
                if allowed:
                    commands=engine.accept(data,frame,frame.source_ns)
                    self.assertEqual(commands['arm_right']['message']['points'][0]['positions'][0],delta)
                else:
                    with self.assertRaisesRegex(SafetyStop,'2 rad/s'):engine.accept(data,frame,frame.source_ns)

    def test_first_segment_checks_held_command_as_well_as_measured_start(self):
        pose=dict(self.frame.positions);pose['arm_r_joint1']=.15  # Still within the 10 degree tracking bound.
        data=chunk(self.startup,self.source)
        for row in data['positions']:row[0]=.15  # Zero distance from measurement but 3 rad/s from the commanded hold.
        with self.assertRaisesRegex(SafetyStop,'preceding hold'):
            self.engine.accept(data,feedback(self.startup.profile,self.source,pose),self.source)

    def test_measured_velocity_limit_checks_both_signs_and_boundary(self):
        for speed in [2.,-2.,2.001,-2.001]:
            frame=feedback(self.startup.profile,self.source,self.frame.positions,{'arm_r_joint1':speed})
            with self.subTest(speed=speed):
                if abs(speed)<=2: self.engine.monitor(frame,self.source)
                else:
                    with self.assertRaisesRegex(SafetyStop,'2 rad/s'):self.engine.monitor(frame,self.source)

    def test_angular_tracking_limit_is_ten_degrees_for_right_and_passive_joints(self):
        for name in ['arm_r_joint1','arm_l_joint1','head_joint1']:
            for degrees in [9.,10.,10.01]:
                pose=dict(self.frame.positions);pose[name]+=math.radians(degrees)
                frame=feedback(self.startup.profile,self.source,pose)
                with self.subTest(name=name,degrees=degrees):
                    if degrees<=10: self.engine.monitor(frame,self.source)
                    else:
                        with self.assertRaisesRegex(SafetyStop,'10 degrees'): self.engine.monitor(frame,self.source)

    def test_tracking_follows_interpolated_target_not_future_final_target(self):
        data=chunk(self.startup,self.source,step=.05)
        self.accept(data,now=self.source)
        now=self.source+75_000_000
        pose=dict(self.frame.positions);pose['arm_r_joint1']=.075
        self.engine.monitor(feedback(self.startup.profile,now,pose),now)
        self.assertAlmostEqual(self.engine.expected_right(now)['arm_r_joint1'],.075)

    def test_angle_limits_are_not_applied_to_prismatic_elevator(self):
        pose=dict(self.frame.positions);pose['lift_joint']=-.3
        self.engine.monitor(feedback(self.startup.profile,self.source,pose,{'lift_joint':3.}),self.source)

    def test_nonfinite_and_out_of_bounds_predictions_are_safety_stops(self):
        for value in [float('nan'),4.]:
            data=chunk(self.startup,self.source);data['positions'][0][0]=value
            with self.subTest(value=value),self.assertRaises(SafetyStop):self.accept(data)

    def test_publish_failure_does_not_commit_chunk(self):
        def fail(_):raise RuntimeError('publish failed')
        with self.assertRaises(RuntimeError):self.accept(submit=fail)
        self.assertIsNone(self.engine.last_commands);self.assertEqual(self.startup.phase,'ready')

    def test_old_protocol_session_duplicate_shape_units_and_future_time_are_rejected(self):
        for updates in [{'execution_contract':'async'}, {'epoch':'old-session'}, {'cycle':9},
                        {'positions':[[0.]*8]*15}, {'units':['degrees']*8},
                        {'joint_names':list(reversed(self.startup.profile['policy']['joint_names']))},
                        {'source_time_ns':self.source+100_000_000}]:
            data={**chunk(self.startup,self.source),**updates}
            with self.subTest(updates=updates),self.assertRaises(BridgeError):self.accept(data)
        self.accept()
        with self.assertRaisesRegex(BridgeError,'Duplicate'):self.accept()

    def test_stop_is_terminal_and_holds_all_fresh_measured_positions(self):
        self.accept();now=self.source+100_000_000
        pose=dict(self.frame.positions);pose['arm_r_joint1']=.003;pose['arm_l_joint1']=.04
        commands=self.engine.stop(feedback(self.startup.profile,now,pose),now,fault=True)
        self.assertEqual(self.startup.phase,'fault');self.assertFalse(self.engine.status(now)['needs_chunk'])
        for group,spec in self.startup.profile['commands'].items():
            self.assertEqual(commands[group]['message']['points'][0]['positions'],[pose[n] for n in spec['joint_names']])
        with self.assertRaises(BridgeError):self.accept(chunk(self.startup,now,index=2),now)

    def test_stale_feedback_and_clock_jump_are_rejected(self):
        with self.assertRaisesRegex(BridgeError,'Stale'):self.engine.tick(self.frame,self.source+200_000_000)
        self.engine.clock(self.source,1_000_000)
        with self.assertRaisesRegex(BridgeError,'clock jumped'):self.engine.clock(self.source+1_000_000_000,51_000_000)


class BundleTests(unittest.TestCase):
    def test_bundle_is_standalone_small_and_passes_offline_check(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'ffw'
            manifest = build(out)
            self.assertLess(sum(v['bytes'] for v in manifest['files'].values()), 200_000)
            self.assertFalse(json.loads((out/'config/profile.reference.json').read_text())['execution_enabled'])
            self.assertFalse(manifest['image_build_required'])
            self.assertFalse((out/'Dockerfile').exists())
            for name, record in manifest['files'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), record['sha256'])
            result = subprocess.run([sys.executable,'-B','-m','cosmos_ffw.node','--profile',str(out/'config/profile.reference.json'),
                      '--initial-positions',str(HOME),'--check'],cwd=out,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)['robot_connection_opened'])
            with self.assertRaises(ValueError):build(out)

    def test_compose_reuses_installed_image_and_mounts_code_read_only_without_bringup(self):
        import yaml
        config = yaml.safe_load((ROOT/'deploy/ffw/compose.yaml').read_text())['services']['companion']
        self.assertNotIn('devices',config)
        self.assertNotIn('privileged',config)
        self.assertTrue(config['read_only'])
        self.assertIn('${COSMOS_MODE:-preview}',config['command'])
        self.assertNotIn('build',config)
        self.assertEqual(config['image'],'${ROBOTIS_IMAGE:-robotis/ai-worker:2.2.8}')
        self.assertEqual(config['pull_policy'],'never')
        self.assertEqual(config['entrypoint'],['/bin/bash','/opt/cosmos_ffw/run.sh'])
        self.assertTrue(config['healthcheck']['disable'])
        mounts={value['target']:value for value in config['volumes']}
        self.assertEqual(mounts['/opt/cosmos_ffw/cosmos_ffw']['source'],'./cosmos_ffw')
        self.assertEqual(mounts['/opt/cosmos_ffw/run.sh']['source'],'./run.sh')
        for mount in mounts.values():
            self.assertTrue(mount['read_only'])
            self.assertFalse(mount['bind']['create_host_path'])


class NodeWiringTests(unittest.TestCase):
    """Mock the ROS graph/service boundary, not the chunk/homing implementation."""
    @classmethod
    def setUpClass(cls):
        import importlib
        cls.folder = tempfile.TemporaryDirectory()
        cls.out = Path(cls.folder.name) / "bundle"
        build(cls.out)
        sys.path.insert(0, str(cls.out))
        cls.module = importlib.import_module("cosmos_ffw.node")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.out))
        for name in list(sys.modules):
            if name == 'cosmos_ffw' or name.startswith('cosmos_ffw.'):
                sys.modules.pop(name)
        cls.folder.cleanup()

    def node(self, mode):
        from types import SimpleNamespace as NS, ModuleType
        from unittest import mock
        class Message:
            def __init__(self, **kw): self.__dict__.update(kw)
        class Trajectory(Message): pass
        class Pub:
            def __init__(self, typ, topic): self.typ,self.topic,self.sent=typ,topic,[]
            def publish(self, msg): self.sent.append(msg)
        class Timer:
            def cancel(self): pass
        class Client:
            def service_is_ready(self): return False
        class FakeNode:
            def __init__(self, name):
                self.name,self.robot_ns,self.pubs,self.foreign=name,BASE,[],set()
                self.show_own=True
            def get_parameter(self, name): return NS(value=False)
            def get_clock(self): return NS(now=lambda:NS(nanoseconds=self.robot_ns))
            def create_publisher(self, typ, topic, qos):
                value=Pub(typ,topic);self.pubs.append(value);return value
            def create_subscription(self,*args):return NS()
            def create_client(self,*args):return Client()
            def create_timer(self,*args):return Timer()
            def get_name(self):return self.name
            def get_namespace(self):return '/'
            def get_publishers_info_by_topic(self, topic):
                result=[NS(node_name=self.name,node_namespace='/') for p in self.pubs if p.topic==topic] if self.show_own else []
                if topic in self.foreign:result.append(NS(node_name='another_policy',node_namespace='/'))
                return result
            def count_subscribers(self,topic):return 1
            def get_logger(self):return NS(error=lambda message:None)
        modules={}
        for name in ['rclpy.node','rclpy.qos','sensor_msgs.msg','std_msgs.msg','trajectory_msgs.msg','action_msgs.msg','controller_manager_msgs','controller_manager_msgs.srv']:
            modules[name]=ModuleType(name)
        modules['rclpy.node'].Node=FakeNode
        modules['rclpy.qos'].qos_profile_sensor_data=NS()
        modules['rclpy.qos'].QoSProfile=lambda **kw:NS(**kw)
        modules['rclpy.qos'].DurabilityPolicy=NS(TRANSIENT_LOCAL=1)
        modules['rclpy.qos'].ReliabilityPolicy=NS(RELIABLE=1)
        modules['sensor_msgs.msg'].JointState=Message
        modules['std_msgs.msg'].String=Message
        modules['trajectory_msgs.msg'].JointTrajectory=Trajectory
        modules['action_msgs.msg'].GoalStatusArray=Message
        modules['controller_manager_msgs.srv'].ListControllers=NS(Request=Message)
        p=profile();home=load_home_yaml(HOME,p)
        with mock.patch.dict(sys.modules,modules):
            node=self.module.make_node(p,home,mode)
        patcher=mock.patch.object(self.module,'joint_trajectory',lambda data:NS(data=data));patcher.start();self.addCleanup(patcher.stop)
        node.trajectory_class=Trajectory
        return node

    def make_ready(self, node):
        from types import SimpleNamespace as NS
        for group in GROUPS:node.startup.homing_status(group,4,BASE)
        for i in range(12):
            now=BASE+i*50_000_000
            frame=feedback(profile(),now)
            node.robot_ns=now
            node.startup.update(frame,now)
            node.feedback=frame
        return node

    def test_preview_creates_no_hardware_publishers_even_after_chunk(self):
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('preview'))
        node.tick()
        node.chunk(NS(data=json.dumps(chunk(node.startup,node.feedback.source_ns))))
        self.assertEqual(node.startup.phase,'running')
        self.assertFalse(any(p.typ is node.trajectory_class for p in node.pubs))
        self.assertEqual(node.command_count,0)

    def test_execute_waits_for_controller_health_before_publisher_creation(self):
        node=self.make_ready(self.node('execute'));node.tick()
        self.assertEqual(node.startup.phase,'ready')
        self.assertFalse(node.command_publishers)

    def test_execute_captures_and_publishes_initial_hold_only_after_guard(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        self.assertTrue(node.initial_hold_sent)
        self.assertEqual(node.command_count,4)
        self.assertEqual(set(node.command_publishers),set(GROUPS))

    def test_graph_discovery_delay_does_not_fault_or_publish(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.show_own=False;node.tick()
        self.assertEqual(node.startup.phase,'ready')
        self.assertEqual(node.command_count,0)
        node.show_own=True;node.tick()
        self.assertEqual(node.command_count,4)

    def test_competing_publisher_prevents_activation(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic()
        node.foreign.add(profile()['commands']['arm_right']['topic']);node.tick()
        self.assertEqual(node.startup.phase,'fault')
        self.assertEqual(node.command_count,0)

    def test_new_external_homing_is_not_preempted_by_a_stop_command(self):
        import time
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count
        stamp=NS(sec=12,nanosec=0)
        node.homing('arm_right',NS(status_list=[NS(status=2,goal_info=NS(stamp=stamp))]))
        self.assertEqual(node.startup.phase,'fault')
        self.assertIsNone(node.startup.hold)
        self.assertEqual(node.command_count,before)

    def test_shutdown_announces_a_valid_stopped_epoch_without_preview_commands(self):
        import re
        node=self.make_ready(self.node('preview'))
        previous=node.startup.epoch
        node.shutdown_hold()
        status=json.loads(node.status_pub.sent[-1].data)
        self.assertEqual(status['phase'],'stopped')
        self.assertNotEqual(status['epoch'],previous)
        self.assertTrue(re.fullmatch(r'[0-9a-f]{32}',status['epoch']))
        self.assertEqual(node.command_count,0)
        before=len(node.status_pub.sent);node.shutdown_hold()
        self.assertEqual(len(node.status_pub.sent),before)

    def test_unsafe_chunk_faults_session_and_sends_stop_instead_of_only_rejecting(self):
        import time
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count
        data=chunk(node.startup,node.feedback.source_ns)
        data['positions'][0][0]=.11
        node.chunk(NS(data=json.dumps(data)))
        self.assertEqual(node.startup.phase,'fault');self.assertIn('2 rad/s',node.reason)
        self.assertEqual(node.command_count,before+4)
        self.assertFalse(node.engine.status(node.robot_ns)['needs_chunk'])
        before=node.command_count
        node.chunk(NS(data=json.dumps(chunk(node.startup,node.feedback.source_ns,index=2))))
        self.assertEqual(node.command_count,before)

    def test_measured_velocity_violation_in_joint_callback_faults_and_stops(self):
        import time
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        names=all_names(profile()); velocities=[0.]*len(names);velocities[names.index('arm_r_joint1')]=2.01
        stamp=NS(sec=node.robot_ns//10**9,nanosec=node.robot_ns%10**9+1)
        before=node.command_count
        node.joints(NS(name=names,position=[node.feedback.positions[n] for n in names],velocity=velocities,header=NS(stamp=stamp)))
        self.assertEqual(node.startup.phase,'fault');self.assertIn('2 rad/s',node.reason)
        self.assertEqual(node.command_count,before+4)

    def test_runtime_tracking_violation_faults_and_stops(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        pose=dict(node.feedback.positions);pose['arm_r_joint1']=math.radians(11)
        node.feedback=feedback(profile(),node.robot_ns,pose)
        before=node.command_count;node.tick()
        self.assertEqual(node.startup.phase,'fault');self.assertIn('10 degrees',node.reason)
        self.assertEqual(node.command_count,before+4)

    def test_feedback_loss_faults_without_inventing_a_stop_position(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count;node.feedback=None;node.tick()
        self.assertEqual(node.startup.phase,'fault')
        self.assertEqual(node.engine.execution_end_ns,0)
        self.assertEqual(node.command_count,before)

    def test_controller_health_loss_faults_and_cannot_publish_a_hold(self):
        import time
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count;node.controller_ok_at=0;node.tick()
        self.assertEqual(node.startup.phase,'fault')
        self.assertEqual(node.engine.execution_end_ns,0)
        self.assertEqual(node.command_count,before)

    def test_stale_lease_and_stale_feedback_cannot_submit_a_chunk(self):
        import time
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count;node.controller_ok_at=0
        node.chunk(NS(data=json.dumps(chunk(node.startup,node.feedback.source_ns))))
        self.assertEqual(node.command_count,before)
        self.assertIsNone(node.engine.last_commands)


if __name__ == '__main__':unittest.main()
