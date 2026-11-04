"""Native ROS 2 FFW companion. No driver startup, model loading, or SSH operations."""
from pathlib import Path
import argparse
import json
import signal
import time
import uuid

from .core import BridgeError, CONTROLLERS, GROUPS, JointFeedback, Startup, hold_messages, load_home_yaml, require, stamp_ns, validate_profile
from .chunks import ChunkExecutor
from .ros_messages import joint_trajectory


def make_node(profile, home, mode, velocity_scale):
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
    from action_msgs.msg import GoalStatusArray
    from controller_manager_msgs.srv import ListControllers

    class Companion(Node):
        def __init__(self):
            super().__init__("cosmos_ffw_companion")
            require(self.get_parameter("use_sim_time").value is False, "Physical executor requires the robot clock, not simulated time")
            self.mode = mode
            self.startup = Startup(profile, home, self.now())
            self.engine = ChunkExecutor(self.startup, velocity_scale=velocity_scale)
            self.feedback, self.command_publishers = None, {}
            self.controller_ok_at, self.pending_query = 0., None
            self.query_started = 0.
            self.reason, self.last_rejection, self.command_count = "Waiting for ROBOTIS homing", None, 0
            self.closing, self.stop_sent = False, False
            self.initial_hold_sent = False
            self.activation_started = None
            topics = profile.get("bridge_topics", {})
            self.status_pub = self.create_publisher(String, topics.get("status", "/cosmos/bg2/status"), 10)
            self.create_subscription(JointState, profile["joint_feedback"]["topic"], self.joints, qos_profile_sensor_data)
            self.create_subscription(String, topics.get("chunk", "/cosmos/bg2/chunk"), self.chunk, 1)
            self.create_subscription(String, topics.get("control", "/cosmos/bg2/control"), self.control, 1)
            status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for group, data in home.items():
                self.create_subscription(GoalStatusArray, data["status_topic"],
                                         lambda msg, group=group: self.homing(group, msg), status_qos)
            self.controller_client = self.create_client(ListControllers, "/controller_manager/list_controllers")
            self.timer = self.create_timer(.05, self.tick)
            self.health_timer = self.create_timer(.25, self.query_controllers)

        def now(self):
            return self.get_clock().now().nanoseconds

        def joints(self, msg):
            try:
                frame = JointFeedback.from_arrays(msg.name, msg.position, msg.velocity,
                                                  stamp_ns(msg.header.stamp), self.now(), profile)
                require(self.feedback is None or frame.source_ns > self.feedback.source_ns, "Joint timestamps did not increase")
                self.feedback = frame
                self.startup.update(frame, self.now())
            except (ValueError, KeyError) as error:
                self.fault(str(error))

        def homing(self, group, msg):
            try:
                for item in msg.status_list:
                    self.startup.homing_status(group, int(item.status), stamp_ns(item.goal_info.stamp))
            except (ValueError, KeyError) as error:
                # Startup clears its captured hold on external reinitialization.
                # Never preempt the new ROBOTIS homing action with old hold commands.
                self.fault(str(error))

        def query_controllers(self):
            if self.closing or self.startup.phase in ("fault", "stopped"):
                return
            if self.pending_query is not None:
                if time.monotonic() - self.query_started > .75:
                    self.controller_ok_at = 0.
                    self.pending_query.cancel()
                    self.pending_query = None
                return
            if not self.controller_client.service_is_ready():
                self.controller_ok_at = 0.
                return
            self.query_started = time.monotonic()
            future = self.controller_client.call_async(ListControllers.Request())
            self.pending_query = future
            future.add_done_callback(self.controller_result)

        def controller_result(self, future):
            if future is not self.pending_query:
                return
            self.pending_query = None
            try:
                response = future.result()
                require(response is not None, "No controller response")
                by_name = {c.name: c for c in response.controller}
                for group, name in CONTROLLERS.items():
                    require(name in by_name and by_name[name].state == "active", f"Controller is not active: {name}")
                    interfaces = {n + "/position" for n in profile["commands"][group]["joint_names"]}
                    require(interfaces <= set(by_name[name].claimed_interfaces), f"Wrong command interfaces: {name}")
                self.controller_ok_at = time.monotonic()
            except Exception as error:
                self.controller_ok_at = 0.
                self.reason = str(error)

        def owners_ok(self, *, allow_missing_self=False):
            for group in GROUPS:
                spec = profile["commands"][group]
                endpoints = self.get_publishers_info_by_topic(spec["topic"])
                foreign = [ep for ep in endpoints if ep.node_name != self.get_name() or ep.node_namespace != self.get_namespace()]
                require(not foreign and len(endpoints) <= 1, f"Competing command publisher: {group}")
                if not allow_missing_self:
                    require(len(endpoints) == 1, f"Own command publisher is not visible: {group}")
                    require(self.count_subscribers(spec["topic"]) >= 1, f"No controller subscriber: {group}")

        def guard(self):
            require(self.controller_ok_at > 0 and time.monotonic() - self.controller_ok_at <= 1., "Controller health lease expired")
            self.owners_ok()

        def activate(self):
            if self.mode != "execute" or self.command_publishers:
                return
            require(self.startup.phase == "ready", "Homing must finish before activation")
            require(self.controller_ok_at > 0 and time.monotonic() - self.controller_ok_at <= 1., "Waiting for active position controllers")
            self.owners_ok(allow_missing_self=True)
            self.activation_started = time.monotonic()
            for group in GROUPS:
                self.command_publishers[group] = self.create_publisher(JointTrajectory, profile["commands"][group]["topic"], 1)

        def send(self, commands):
            if self.mode != "execute":
                return
            self.guard()
            # Convert all groups before publishing any; use the existing JTC to
            # buffer/interpolate the complete trajectory at its own update rate.
            prepared = {group: joint_trajectory(bundle["message"]) for group, bundle in commands.items()}
            for group in ("arm_left", "head", "lift", "arm_right"):
                if group in prepared:
                    self.command_publishers[group].publish(prepared[group])
                    self.command_count += 1

        def chunk(self, msg):
            try:
                require(not self.closing, "Companion is stopping")
                require(len(msg.data.encode()) <= 128 * 1024, "Oversized chunk")
                if self.mode == "execute":
                    self.guard()
                require(self.feedback is not None, "Waiting for feedback")
                data = json.loads(msg.data)
                self.engine.accept(data, self.feedback, self.now(), submit=self.send)
                self.stop_sent = False
                self.reason, self.last_rejection = "Chunk accepted", None
            except (ValueError, KeyError, TypeError) as error:
                self.last_rejection = str(error)
            except Exception as error:
                self.fault("Publishing failed: " + str(error))

        def control(self, msg):
            try:
                require(len(msg.data.encode()) <= 1024, "Oversized control message")
                data = json.loads(msg.data)
                require(data.get("schema") == "cosmos_bg2_control_v1" and data.get("epoch") == self.startup.epoch,
                        "Control message session mismatch")
                require(data.get("command") == "stop", "Only stop is supported; restart and home for a new session")
                self.shutdown_hold()
            except (ValueError, KeyError, TypeError) as error:
                self.last_rejection = str(error)

        def fault(self, reason):
            if self.startup.phase == "fault" and self.reason == reason:
                return
            self.reason = reason
            try:
                commands = self.engine.stop(self.feedback, self.now(), fault=True)
                if commands is not None and self.command_publishers and not self.stop_sent:
                    self.send(commands)
                    self.stop_sent = True
            except Exception:
                # Missing feedback/ownership must not create invented stop positions.
                pass
            self.startup.phase = "fault"
            self.engine.valid_until_ns = 0
            self.get_logger().error(reason)

        def tick(self):
            if self.closing:
                return
            try:
                self.engine.clock(self.now(), time.monotonic_ns())
                if self.startup.phase in ("ready", "running", "holding"):
                    require(self.feedback is not None, "No feedback")
                    can_tick = True
                    if self.mode == "execute":
                        if self.startup.phase == "ready" and not self.command_publishers and not (
                                self.controller_ok_at > 0 and time.monotonic() - self.controller_ok_at <= 1.):
                            self.reason = "Waiting for active position controllers"
                            can_tick = False
                        else:
                            self.activate()
                            self.owners_ok(allow_missing_self=True)
                            visible = all(len(self.get_publishers_info_by_topic(profile["commands"][g]["topic"])) == 1 and
                                          self.count_subscribers(profile["commands"][g]["topic"]) >= 1 for g in GROUPS)
                            if not visible and not self.initial_hold_sent:
                                require(time.monotonic() - self.activation_started < 1., "Controller endpoints did not appear")
                                self.reason, can_tick = "Waiting for command endpoint discovery", False
                            else:
                                self.guard()
                                if not self.initial_hold_sent:
                                    self.send(hold_messages(self.feedback, self.startup, self.now(), include_right=True))
                                    self.initial_hold_sent = True
                    if can_tick:
                        commands = self.engine.tick(self.feedback, self.now(), check_tracking=self.mode == "execute")
                        if commands is not None:
                            self.send(commands)
                            self.reason = "Chunk ended; holding and waiting for a fresh prediction"
            except (ValueError, KeyError) as error:
                self.fault(str(error))
            status = self.engine.status(self.now())
            status.update(mode=self.mode, physical_publishing=self.mode == "execute", command_messages=self.command_count,
                          reason=self.reason, last_rejection=self.last_rejection,
                          controllers_ready=self.controller_ok_at > 0 and time.monotonic() - self.controller_ok_at <= 1.)
            self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))

        def shutdown_hold(self):
            if self.closing:
                return
            try:
                commands = self.engine.stop(self.feedback, self.now())
                if commands is not None and self.command_publishers:
                    self.send(commands)
            except Exception as error:
                self.get_logger().error("Could not send shutdown hold: " + str(error))
            self.closing = True
            self.startup.stop()
            self.startup.epoch = uuid.uuid4().hex
            status = self.engine.status(self.now())
            status.update(mode=self.mode, physical_publishing=self.mode == "execute",
                          command_messages=self.command_count, reason="Companion stopped", controllers_ready=False)
            self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
            self.timer.cancel()
            self.health_timer.cancel()

    return Companion()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--initial-positions", type=Path, required=True)
    parser.add_argument("--mode", choices=("preview", "execute"), default="preview")
    parser.add_argument("--velocity-scale", type=float, default=.25)
    parser.add_argument("--check", action="store_true", help="Validate files only; do not import/init ROS or connect")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    validate_profile(profile, live=args.mode == "execute", execute=args.mode == "execute")
    require(0 < args.velocity_scale <= 1, "Invalid velocity scale")
    home = load_home_yaml(args.initial_positions, profile)
    if args.check:
        print(json.dumps({"valid": True, "mode": args.mode, "homing_groups": sorted(home), "robot_connection_opened": False}))
        return
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        node = make_node(profile, home, args.mode, args.velocity_scale)
        while not stopping and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        if node is not None:
            node.shutdown_hold()
            rclpy.spin_once(node, timeout_sec=.05)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
