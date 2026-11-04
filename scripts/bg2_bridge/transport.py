"""ROBOTIS Zenoh adapter. Observing creates no publishers; activation is explicit."""
from pathlib import Path
import os
import sys
import time
import uuid

import numpy as np

from .core import BridgeError, CONTROLLERS, GROUPS, JointFeedback, require, stamp_ns, validate_profile


class ZenohTransport:
    def __init__(self, profile, observations, startup=None, *, sdk=None):
        self.profile, self.observations, self.startup = profile, observations, startup
        self.publishers, self.subscribers, self.controller_client = {}, [], None
        self.node_name = "cosmos_bg2_" + uuid.uuid4().hex[:12]
        self.guard_checked_at = 0.
        self.guard_error = "Controller ownership has not been checked"
        middleware = profile["middleware"]
        self.options = {"domain_id": middleware["domain_id"], "router_ip": middleware["configured_router_host"],
                        "router_port": middleware["configured_router_port"]}
        if sdk is None:
            sdk_root = Path(__file__).resolve().parents[2] / "third_party/zenoh_ros2_sdk"
            sys.path.insert(0, str(sdk_root))
            # Specify client-only transport before the SDK's first session is created.
            import json
            endpoint = f"tcp/{self.options['router_ip']}:{self.options['router_port']}"
            os.environ["ZENOH_CONFIG_OVERRIDE"] = ('mode="client";connect/endpoints=' + json.dumps([endpoint]) +
                       ';scouting/multicast/enabled=false;transport/shared_memory/enabled=false')
            import zenoh_ros2_sdk as sdk
            require(sdk.__version__ == "0.1.8", "Use the pinned ROBOTIS SDK")
        self.sdk = sdk
        try:
            feedback = profile["joint_feedback"]
            self._subscribe(feedback["topic"], feedback["msg_type"], self._joints)
            for key, spec in profile["cameras"].items():
                self._subscribe(spec["topic"], spec["msg_type"], lambda msg, key=key: self._camera(key, msg))
            if startup is not None:
                for group, spec in startup.home.items():
                    self._subscribe(spec["status_topic"], "action_msgs/msg/GoalStatusArray",
                                    lambda msg, group=group: self._homing(group, msg))
        except Exception:
            self.close()
            raise

    def _subscribe(self, topic, msg_type, callback):
        def checked(msg):
            try:
                callback(msg)
            except Exception as error:
                with self.observations.lock:
                    self.observations.error = str(error)
        self.subscribers.append(self.sdk.ROS2Subscriber(topic=topic, msg_type=msg_type, callback=checked,
                                                       node_name=self.node_name, **self.options))

    def _joints(self, msg):
        frame = JointFeedback.from_arrays(msg.name, msg.position, msg.velocity, stamp_ns(msg.header.stamp), time.time_ns(), self.profile)
        self.observations.add_joints(frame)

    def _camera(self, key, msg):
        self.observations.add_camera(key, stamp_ns(msg.header.stamp), time.time_ns(), bytes(msg.data))

    def _homing(self, group, msg):
        with self.observations.lock:
            for status in msg.status_list:
                self.startup.homing_status(group, int(status.status), stamp_ns(status.goal_info.stamp))

    def activate(self):
        validate_profile(self.profile, live=True, execute=True)
        require(self.startup is not None and self.startup.phase == "ready", "Homing/settling must complete before publisher creation")
        try:
            self.controller_client = self.sdk.ROS2ServiceClient(service_name="/controller_manager/list_controllers",
                    srv_type="controller_manager_msgs/srv/ListControllers", timeout=0.5,
                    node_name=self.node_name, **self.options)
            for group in GROUPS:
                spec = self.profile["commands"][group]
                self.publishers[group] = self.sdk.ROS2Publisher(topic=spec["topic"], msg_type=spec["msg_type"],
                                                               node_name=self.node_name, **self.options)
            self.check_ownership()
        except Exception:
            self.close_publishers()
            raise

    def check_ownership(self):
        self.guard_error = "Controller ownership check in progress"
        try:
            response = self.controller_client.call()
            require(response is not None, "Controller-manager query timed out")
            controllers = {c.name: c for c in response.controller}
            for group, name in CONTROLLERS.items():
                require(name in controllers and controllers[name].state == "active", f"Controller is not active: {name}")
                required = {n + "/position" for n in self.profile["commands"][group]["joint_names"]}
                require(required <= set(controllers[name].claimed_interfaces), f"Position interfaces are not owned by {name}")
                info = self.sdk.get_topic_info(self.profile["commands"][group]["topic"], timeout=0.05, verbose=True, **self.options)
                require(info is not None and info.publisher_count == 1 and info.subscriber_count >= 1 and
                        len(info.publishers) == 1 and info.publishers[0].node_name == self.node_name,
                        f"Missing controller subscriber or competing command publisher: {group}")
            self.guard_checked_at, self.guard_error = time.monotonic(), None
        except Exception as error:
            self.guard_error = str(error)
            raise

    def publish(self, bundles):
        require(self.guard_error is None and time.monotonic() - self.guard_checked_at <= 1., "Controller ownership check is stale/failed")
        # Convert every message before putting any command on the network.
        classes = {name: self.sdk.get_message_class(name) for name in
                   ("builtin_interfaces/msg/Time", "builtin_interfaces/msg/Duration", "std_msgs/msg/Header", "trajectory_msgs/msg/JointTrajectoryPoint")}
        require(all(classes.values()), "Missing ROS message definitions")
        converted = {}
        for group, bundle in bundles.items():
            require(bundle["topic"] == self.profile["commands"][group]["topic"], "Command destination changed")
            msg = bundle["message"]
            header = classes["std_msgs/msg/Header"](stamp=classes["builtin_interfaces/msg/Time"](**msg["header"]["stamp"]), frame_id="")
            points = [classes["trajectory_msgs/msg/JointTrajectoryPoint"](
                      positions=np.asarray(point["positions"], dtype=np.float64), velocities=np.array([], dtype=np.float64),
                      accelerations=np.array([], dtype=np.float64), effort=np.array([], dtype=np.float64),
                      time_from_start=classes["builtin_interfaces/msg/Duration"](**point["time_from_start"])) for point in msg["points"]]
            converted[group] = {"header": header, "joint_names": msg["joint_names"], "points": points}
        for group in ("arm_left", "head", "lift", "arm_right"):
            if group in converted:
                self.publishers[group].publish(**converted[group])

    def close_publishers(self):
        for publisher in self.publishers.values():
            publisher.close()
        self.publishers.clear()
        if self.controller_client is not None:
            self.controller_client.close()
            self.controller_client = None
        self.guard_error = "Transport closed"

    def close(self):
        self.close_publishers()
        for sub in self.subscribers:
            sub.close()
        self.subscribers.clear()
