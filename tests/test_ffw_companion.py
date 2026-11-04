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
from bg2_bridge.core import BridgeError, GROUPS, JointFeedback, Startup, all_names, load_home_yaml, validate_profile
from bg2_bridge.chunks import ChunkExecutor, RequestPlanner, SCHEMA
from package_ffw import build

HOME = ROOT / "third_party/ai_worker/ffw_bringup/config/ffw_bg2_rev4_follower/ffw_bg2_follower_initial_positions.yaml"
BASE = 10_000_000_000


def profile():
    return json.loads((ROOT / "profiles/robotis_bg2_rev4.reference.json").read_text())


def feedback(p, now, positions=None):
    names = all_names(p)
    positions = positions or {name: 0. for name in names}
    return JointFeedback.from_arrays(names, [positions[n] for n in names], [0.] * len(names), now, now, p)


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


def chunk(startup, source, index=1, step=.001):
    right = startup.profile["policy"]["joint_names"]
    rows = [[startup.hold[n] for n in right] for _ in range(16)]
    for i, row in enumerate(rows):
        row[0] += step * (i + 1)
    return {"schema": SCHEMA, "epoch": startup.epoch, "request_id": f"{index:032x}",
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

    def test_reference_profile_cannot_enable_execution(self):
        with self.assertRaises(BridgeError):
            validate_profile(profile(), execute=True)


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.startup, self.frame = ready()
        self.engine = ChunkExecutor(self.startup)
        self.source = self.frame.source_ns

    def accept(self, data=None, now=None, submit=None):
        now = now if now is not None else self.source + 20_000_000
        return self.engine.accept(data or chunk(self.startup, self.source), feedback(self.startup.profile, now, self.frame.positions), now, submit=submit)

    def test_full_chunk_preserves_original_due_times(self):
        commands = self.accept()
        msg = commands["arm_right"]["message"]
        start = msg["header"]["stamp"]["sec"] * 10**9 + msg["header"]["stamp"]["nanosec"]
        due = [start + p["time_from_start"]["sec"] * 10**9 + p["time_from_start"]["nanosec"] for p in msg["points"][1:]]
        self.assertEqual(due[-1], self.source + 800_000_000)
        self.assertTrue(all((v - self.source) % 50_000_000 == 0 for v in due))
        self.assertTrue(all(v > self.source + 20_000_000 for v in due))

    def test_new_chunk_replaces_remaining_tail_without_a_fifo(self):
        old = self.accept()
        next_source = self.source + 100_000_000
        data = chunk(self.startup, next_source, index=2)
        self.accept(data, now=next_source + 20_000_000)
        self.assertEqual(self.engine.last_request_id, data["request_id"])
        self.assertIsNot(self.engine.last_commands, old)
        self.assertEqual(self.engine.valid_until_ns, next_source + 800_000_000)

    def test_partial_expiry_skips_elapsed_rows_without_retiming(self):
        commands = self.accept(now=self.source + 600_000_000)
        self.assertLess(len(commands["arm_right"]["message"]["points"]), 16)
        self.assertEqual(self.engine.valid_until_ns, self.source + 800_000_000)

    def test_expiry_holds_right_and_keeps_other_groups_at_captured_pose(self):
        self.accept()
        now = self.source + 800_000_000
        positions = copy.deepcopy(self.frame.positions); positions["arm_r_joint1"] = .016
        frame = feedback(self.startup.profile, now, positions)
        result = self.engine.tick(frame, now)
        self.assertEqual(self.startup.phase, "holding")
        self.assertEqual(self.engine.valid_until_ns, 0)
        self.assertEqual(result["arm_right"]["message"]["points"][0]["positions"][0], .016)
        for group in self.startup.profile["hold_groups"]:
            names = self.startup.profile["commands"][group]["joint_names"]
            self.assertEqual(result[group]["message"]["points"][0]["positions"], [self.startup.hold[n] for n in names])
        self.assertIsNone(self.engine.tick(frame, now + 1))

    def test_right_hold_drift_after_expiry_is_detected(self):
        self.accept()
        now = self.source + 800_000_000
        self.engine.tick(feedback(self.startup.profile, now), now)
        pose = dict(self.frame.positions); pose['arm_r_joint1'] = .4
        with self.assertRaisesRegex(BridgeError, "Right-arm hold"):
            self.engine.tick(feedback(self.startup.profile, now + 50_000_000, pose), now + 50_000_000)

    def test_fresh_chunk_can_resume_after_transient_starvation(self):
        self.accept()
        now = self.source + 800_000_000
        self.engine.tick(feedback(self.startup.profile, now), now)
        self.accept(chunk(self.startup, now, index=2), now=now)
        self.assertEqual(self.startup.phase, "running")

    def test_entirely_expired_chunk_is_rejected(self):
        with self.assertRaisesRegex(BridgeError, "expired"):
            self.accept(now=self.source + 800_000_000)
        self.assertEqual(self.engine.last_source_ns, 0)

    def test_bad_reply_leaves_previous_chunk_unchanged(self):
        self.accept(); previous = self.engine.last_commands
        bad = chunk(self.startup, self.source + 50_000_000, index=2)
        bad["positions"][3][0] = float('nan')
        with self.assertRaises(BridgeError): self.accept(bad)
        self.assertIs(self.engine.last_commands, previous)
        self.assertEqual(self.engine.last_source_ns, self.source)

    def test_publish_failure_does_not_commit_chunk(self):
        def fail(_): raise RuntimeError("publish failed")
        with self.assertRaises(RuntimeError): self.accept(submit=fail)
        self.assertIsNone(self.engine.last_commands)
        self.assertEqual(self.startup.phase, "ready")

    def test_duplicate_old_session_and_out_of_order_replies_are_rejected(self):
        self.accept()
        with self.assertRaisesRegex(BridgeError, "Duplicate"): self.accept()
        bad = chunk(self.startup, self.source + 50_000_000, index=2);bad["epoch"] = "old-session"
        with self.assertRaisesRegex(BridgeError, "session"):self.accept(bad)
        bad = chunk(self.startup, self.source - 1, index=3)
        with self.assertRaisesRegex(BridgeError, "out-of-order"):self.accept(bad)

    def test_wrong_order_units_shape_and_future_stamp_are_rejected(self):
        for key, value in [("joint_names", list(reversed(self.startup.profile["policy"]["joint_names"]))),
                           ("units", ["degrees"] * 8), ("positions", [[0.] * 8] * 15),
                           ("source_time_ns", self.source + 100_000_000)]:
            bad = chunk(self.startup, self.source);bad[key] = value
            with self.subTest(key=key), self.assertRaises(BridgeError):self.accept(bad)

    def test_position_and_velocity_limits_are_enforced(self):
        for value in [4., .8]:
            bad = chunk(self.startup, self.source)
            for row in bad["positions"]:row[0] = value
            with self.subTest(value=value), self.assertRaises(BridgeError):self.accept(bad)

    def test_stale_feedback_and_hold_drift_are_rejected(self):
        self.accept()
        with self.assertRaisesRegex(BridgeError, "Stale"):
            self.engine.tick(self.frame, self.source + 200_000_000)
        pose = dict(self.frame.positions);pose["arm_l_joint2"] = .3
        with self.assertRaisesRegex(BridgeError, "Held group"):
            self.engine.tick(feedback(self.startup.profile, self.source, pose), self.source)

    def test_right_tracking_failure_is_detected(self):
        self.accept()
        pose = dict(self.frame.positions);pose["arm_r_joint1"] = .5
        now = self.source + 100_000_000
        with self.assertRaisesRegex(BridgeError, "tracking"):
            self.engine.tick(feedback(self.startup.profile, now, pose), now)

    def test_clock_jump_is_rejected(self):
        self.engine.clock(self.source, 1_000_000)
        with self.assertRaisesRegex(BridgeError, "clock jumped"):
            self.engine.clock(self.source + 1_000_000_000, 51_000_000)


class PrefetchTests(unittest.TestCase):
    def test_requests_start_before_chunk_ends_and_only_one_is_in_flight(self):
        planner = RequestPlanner(150_000_000, margin_ns=50_000_000)
        status = {"phase": "running", "remaining_ns": 200_000_000}
        self.assertTrue(planner.should_request(status, 100, 100, 123))
        planner.started(123)
        self.assertFalse(planner.should_request(status, 100, 100, 124))
        planner.finished(160_000_000)
        self.assertTrue(planner.should_request(status, 100, 100, 124))
        self.assertFalse(planner.should_request(status, 100, 100, 123))

    def test_ready_holding_and_stale_status(self):
        planner = RequestPlanner(100_000_000)
        for phase in ['ready','holding']:
            self.assertTrue(planner.should_request({'phase':phase,'remaining_ns':0}, 100, 100, 123))
        self.assertFalse(planner.should_request({'phase':'running','remaining_ns':0},100,200_000_101,123))
        self.assertFalse(planner.should_request({'phase':'fault','remaining_ns':0},100,100,123))

    def test_repeated_fast_requests_overlap_chunks_without_starvation(self):
        startup, frame = ready()
        engine = ChunkExecutor(startup)
        planner = RequestPlanner(150_000_000, margin_ns=50_000_000)
        pending = None
        accepted, held = 0, 0
        source = frame.source_ns
        for step in range(100):
            now = source + step * 50_000_000
            current = feedback(startup.profile, now)
            result = engine.tick(current, now)
            held += int(result is not None)
            if pending is not None and now >= pending[0]:
                _, observed, identifier = pending
                engine.accept(chunk(startup, observed, index=identifier), current, now)
                planner.finished(100_000_000)
                pending = None
                accepted += 1
            status = engine.status(now)
            if planner.should_request(status, step * 50_000_000, step * 50_000_000, now):
                planner.started(now)
                pending = (now + 100_000_000, now, accepted + 1)
        self.assertGreater(accepted, 5)
        self.assertEqual(held, 0)
        self.assertEqual(startup.phase, 'running')

    def test_prefetch_uses_local_elapsed_time_without_server_wall_clock(self):
        planner = RequestPlanner(100_000_000, margin_ns=20_000_000)
        self.assertTrue(planner.should_request({'phase':'running','remaining_ns':200_000_000},100,100_000_100,123))


class BundleTests(unittest.TestCase):
    def test_bundle_is_standalone_small_and_passes_offline_check(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'ffw'
            manifest = build(out)
            self.assertLess(sum(v['bytes'] for v in manifest['files'].values()), 200_000)
            self.assertFalse(json.loads((out/'config/profile.reference.json').read_text())['execution_enabled'])
            for name, record in manifest['files'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), record['sha256'])
            result = subprocess.run([sys.executable,'-B','-m','cosmos_ffw.node','--profile',str(out/'config/profile.reference.json'),
                      '--initial-positions',str(HOME),'--check'],cwd=out,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)['robot_connection_opened'])
            with self.assertRaises(ValueError):build(out)

    def test_compose_has_no_devices_and_defaults_to_preview(self):
        import yaml
        config = yaml.safe_load((ROOT/'deploy/ffw/compose.yaml').read_text())['services']['companion']
        self.assertNotIn('devices',config)
        self.assertNotIn('privileged',config)
        self.assertTrue(config['read_only'])
        self.assertIn('${COSMOS_MODE:-preview}',config['command'])
        self.assertTrue(all(value.endswith(':ro') for value in config['volumes']))


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
            node=self.module.make_node(p,home,mode,.25)
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

    def test_stale_lease_and_stale_feedback_cannot_submit_a_chunk(self):
        import time
        from types import SimpleNamespace as NS
        node=self.make_ready(self.node('execute'));node.controller_ok_at=time.monotonic();node.tick()
        before=node.command_count;node.controller_ok_at=0
        node.chunk(NS(data=json.dumps(chunk(node.startup,node.feedback.source_ns))))
        self.assertEqual(node.command_count,before)
        self.assertIsNone(node.engine.last_commands)


if __name__ == '__main__':unittest.main()
