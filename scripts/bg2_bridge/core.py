"""Transport-independent joint feedback, homing, and timestamped trajectory construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import uuid

import numpy as np

SG2 = "ffw_sg2_rev1"
GROUPS = ("arm_right", "arm_left", "head", "lift")
CONTROLLERS = {"arm_right": "arm_r_controller", "arm_left": "arm_l_controller",
               "head": "head_controller", "lift": "lift_controller"}


class BridgeError(ValueError):
    pass


class SafetyStop(BridgeError):
    """A motion-limit violation that must latch the session in fault."""


MAX_ANGULAR_VELOCITY = 9999.0  # rad/s, explicitly requested by the operator
MAX_ANGULAR_ERROR = math.radians(10)


def angular_names(profile):
    return [n for n in all_names(profile)
            if profile["reference_limits"][n]["type"] in ("revolute", "continuous")]


def safety_require(condition, message):
    if not condition:
        raise SafetyStop(message)


def require(condition, message):
    if not condition:
        raise BridgeError(message)


def stamp_ns(stamp):
    sec, nsec = int(stamp.sec), int(stamp.nanosec)
    require(sec >= 0 and 0 <= nsec < 1_000_000_000, "Invalid ROS timestamp")
    return sec * 1_000_000_000 + nsec


def ros_time(value):
    require(value >= 0, "Negative ROS timestamp/duration")
    sec, nsec = divmod(int(value), 1_000_000_000)
    return {"sec": sec, "nanosec": nsec}


def all_names(profile):
    return [name for group in command_groups(profile) for name in profile["commands"][group]["joint_names"]]


def command_groups(profile):
    return ("arm_left", "arm_right") if profile["robot"] == SG2 else GROUPS


def policy_groups(profile):
    return ("arm_left", "arm_right") if profile["robot"] == SG2 else ("arm_right",)


def validate_profile(profile, *, live=False, execute=False, companion=False):
    sg2 = profile.get("robot") == SG2
    require(profile.get("schema") == 1 and (profile.get("robot") == "ffw_bg2_rev4" or (companion and sg2)),
            "Unsupported robot profile")
    groups = command_groups(profile)
    require(set(profile["commands"]) == set(groups), "Unexpected command groups")
    names = all_names(profile)
    if sg2:
        for side in ("left", "right"):
            spec = profile["commands"]["arm_" + side]
            letter = side[0]
            expected = [f"arm_{letter}_joint{i}" for i in range(1, 8)] + [f"gripper_{letter}_joint1"]
            require(spec["joint_names"] == expected, f"Invalid SG2 {side} controller joints")
            require(spec["topic"] in (f"/leader/joint_trajectory_command_broadcaster_{side}/joint_trajectory",
                                      f"/arm_{letter}_controller/joint_trajectory"), f"Invalid SG2 {side} arm topic")
        require(profile["hold_groups"] == [], "Both SG2 arms must be active")
        require(profile["policy"]["joint_names"] == names, "Expected both SG2 arm groups")
        require(profile["bridge_topics"] == {k: "/cosmos/sg2/" + k for k in ("status", "chunk", "control")},
                "Invalid SG2 bridge topics")
        require(profile["joint_feedback"] == {"topic": "/joint_states", "msg_type": "sensor_msgs/msg/JointState"},
                "Invalid SG2 joint state interface")
    else:
        require(len(names) == len(set(names)) == 19, "Expected 19 unique commanded joints")
        expected = [f"arm_r_joint{i}" for i in range(1, 8)] + ["gripper_r_joint1"]
        require(profile["commands"]["arm_right"]["joint_names"] == profile["policy"]["joint_names"] == expected,
                "Right-arm joint order must match the checkpoint")
    require(profile["policy"]["future_steps"] == 16 and profile["policy"]["frame_interval_ns"] == 50_000_000,
            "Expected the 16-step, 20 Hz companion protocol")
    for name in names:
        limit = profile["reference_limits"][name]
        if sg2:
            require(limit["type"] == "revolute", f"Expected SG2 angular joint: {name}")
        require(all(math.isfinite(limit[k]) for k in ("lower", "upper", "velocity")) and
                limit["lower"] < limit["upper"] and limit["velocity"] > 0, f"Invalid limits for {name}")
    for group in groups:
        command = profile["commands"][group]
        require(command["msg_type"] == "trajectory_msgs/msg/JointTrajectory" and command["topic"].startswith("/"),
                "Expected an absolute JointTrajectory topic")
    require(len({profile["commands"][g]["topic"] for g in groups}) == len(groups), "Command topics must be distinct")
    if not sg2:
        require(set(profile["cameras"]) == {"head", "wrist", "wrist_left"}, "Three camera views are required")
    if live and not sg2:
        for key, camera in profile["cameras"].items():
            require(camera.get("live_to_training_rotation_deg") in (0, 90, 180, 270), f"{key}: verify camera rotation first")
            wh = camera.get("raw_wh")
            require(camera.get("raw_dimensions_verified") is True and isinstance(wh, list) and len(wh) == 2 and
                    all(isinstance(v, int) and v > 0 for v in wh), f"{key}: verify raw_wh first")
    if execute:
        require(profile.get("execution_enabled") is True and profile.get("status") in ("verified_on_robot", "operator_enabled_for_execution"),
                "Execution requires an explicitly enabled deployment profile; reference profiles cannot command hardware")


@dataclass(frozen=True)
class JointFeedback:
    source_ns: int
    received_ns: int
    positions: dict
    velocities: dict

    @classmethod
    def from_arrays(cls, names, positions, velocities, source_ns, received_ns, profile):
        names = list(names)
        require(len(names) == len(set(names)), "Duplicate joint names")
        require(len(names) == len(positions) == len(velocities), "Joint names, positions, velocities must align")
        q, v = dict(zip(names, map(float, positions))), dict(zip(names, map(float, velocities)))
        require(set(all_names(profile)) <= q.keys(), "Incomplete joint feedback")
        if profile["robot"] == SG2:
            arms = all_names(profile)
            q, v = {n: q[n] for n in arms}, {n: v[n] for n in arms}
        require(all(math.isfinite(x) for x in [*q.values(), *v.values()]), "Nonfinite joint feedback")
        require(source_ns > 0 and received_ns > 0, "Joint timestamps must be positive")
        return cls(int(source_ns), int(received_ns), q, v)

    def fresh(self, now_ns, max_age_ns=100_000_000, *, receipt_now_ns=None):
        receipt_now_ns = now_ns if receipt_now_ns is None else receipt_now_ns
        require(0 <= receipt_now_ns - self.received_ns <= max_age_ns, "Stale feedback receipt")
        require(-25_000_000 <= now_ns - self.source_ns <= max_age_ns, "Stale feedback or unsynchronized robot clock")


def load_home_yaml(path: Path, profile):
    """ROBOTIS files repeat the /** root. Preserve every executor instead of dropping earlier ones."""
    import yaml

    class HomeLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            value = loader.construct_object(value_node, deep=True)
            if key in result:
                require(key == "/**" and isinstance(value, dict) and isinstance(result[key], dict), "Duplicate homing YAML key")
                require(not (set(result[key]) & set(value)), "Duplicate homing executor")
                result[key].update(value)
            else:
                result[key] = value
        return result

    HomeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    document = yaml.load(path.read_text(), Loader=HomeLoader)
    executors = document.get("/**", document)
    targets = {}
    for group in command_groups(profile):
        controller = CONTROLLERS[group]
        key = controller.replace("_controller", "_joint_trajectory_executor")
        data = executors[key]["ros__parameters"]
        names = data["joint_names"]
        require(len(names) == len(set(names)) and set(names) == set(profile["commands"][group]["joint_names"]), f"Homing joint mismatch: {group}")
        positions = data[data["step_names"][-1]]
        require(len(names) == len(positions), "Homing pose shape mismatch")
        require(data["action_topic"] == f"/{controller}/follow_joint_trajectory", "Invalid homing action")
        pose = dict(zip(names, map(float, positions)))
        for name, value in pose.items():
            require(math.isfinite(value), f"Nonfinite homing position: {name}")
        targets[group] = {"positions": pose, "position_tolerance": float(data["position_tolerance"]),
                          "velocity_tolerance": float(data["velocity_tolerance"]),
                          "status_topic": data["action_topic"] + "/_action/status"}
        require(all(math.isfinite(targets[group][k]) and targets[group][k] >= 0 for k in ("position_tolerance", "velocity_tolerance")),
                "Invalid homing tolerances")
        # ROS action transport uses /<action-name>/_action/status.
    return targets


class Startup:
    def __init__(self, profile, home, started_ns, settle_ns=500_000_000, *, require_homing=True):
        self.profile, self.home, self.started_ns, self.settle_ns = profile, home, started_ns, settle_ns
        self.require_homing = require_homing
        self.phase, self.epoch = "initializing" if require_homing else "settling", uuid.uuid4().hex
        self.current_anchor = None
        self.completed, self.hold = set(), None
        self.goal_starts = {}
        self.stable_since = self.last_feedback_ns = None

    def homing_status(self, group, status, goal_started_ns):
        require(group in command_groups(self.profile), "Unknown homing group")
        # Current-pose mode does not infer homing from controller action statuses.
        if not self.require_homing:
            return
        if goal_started_ns < self.started_ns:
            return
        if self.phase in ("ready", "running", "holding", "paused"):
            if goal_started_ns > self.goal_starts.get(group, -1):
                # An external homing action now owns the controllers. Do not send a
                # stop/hold trajectory that would preempt that action.
                self.phase, self.hold, self.epoch = "fault", None, uuid.uuid4().hex
                raise BridgeError("New homing started; discard this inference session")
            return
        if self.phase not in ("initializing", "settling"):
            return
        if goal_started_ns < self.goal_starts.get(group, -1):
            return
        if goal_started_ns > self.goal_starts.get(group, -1):
            self.completed.discard(group)
            self.goal_starts[group] = goal_started_ns
            self.phase, self.stable_since = "initializing", None
        if status in (5, 6):
            self.phase = "fault"
            raise BridgeError(f"Homing failed: {group}")
        if status == 4:
            self.completed.add(group)
            if self.completed == set(command_groups(self.profile)):
                self.phase = "settling"

    def update(self, feedback, now_ns):
        feedback.fresh(now_ns)
        if self.phase != "settling":
            return
        require(self.last_feedback_ns is None or feedback.source_ns >= self.last_feedback_ns, "Joint time moved backwards")
        if self.last_feedback_ns is not None and feedback.source_ns - self.last_feedback_ns > 100_000_000:
            self.stable_since = None
        if self.last_feedback_ns == feedback.source_ns:
            return
        self.last_feedback_ns = feedback.source_ns
        if not self.require_homing:
            for group in command_groups(self.profile):
                names = self.profile["commands"][group]["joint_names"]
                check_positions(names, [feedback.positions[n] for n in names], self.profile)
            if self.current_anchor is None:
                self.current_anchor = dict(feedback.positions)
        settled = True
        for data in self.home.values():
            for name, target in data["positions"].items():
                anchor = target if self.require_homing else self.current_anchor[name]
                # Current-pose startup uses the operator's existing angular limit,
                # not the much tighter homing velocity tolerance on noisy grippers.
                angular = self.profile["reference_limits"][name]["type"] in ("revolute", "continuous")
                velocity_limit = data["velocity_tolerance"]
                if self.profile["robot"] != SG2 and not self.require_homing and angular:
                    velocity_limit = MAX_ANGULAR_VELOCITY
                if (abs(feedback.positions[name] - anchor) > data["position_tolerance"] or
                        abs(feedback.velocities[name]) > velocity_limit):
                    settled = False
                    break
            if not settled:
                break
        if not settled:
            self.stable_since = None
            if not self.require_homing:
                self.current_anchor = dict(feedback.positions)
            return
        if self.stable_since is None:
            self.stable_since = feedback.source_ns
        if feedback.source_ns - self.stable_since >= self.settle_ns:
            self.hold = copy.deepcopy(feedback.positions)
            self.phase = "ready"

    def stop(self):
        self.phase = "stopped"


def check_positions(names, values, profile):
    require(len(names) == len(values), "Position vector shape mismatch")
    # Measured holds use the actual coordinates. SG2 target limits are checked in trajectories().
    for name, value in zip(names, values):
        require(math.isfinite(float(value)), f"Nonfinite joint position: {name}")


def message(names, rows, offsets, start_ns):
    return {"header": {"stamp": ros_time(start_ns), "frame_id": ""}, "joint_names": list(names),
            "points": [{"positions": list(map(float, row)), "velocities": [], "accelerations": [], "effort": [],
                        "time_from_start": ros_time(offset)} for row, offset in zip(rows, offsets)]}


def trajectories(prediction, feedback, startup, now_ns):
    """Send all 16 unchanged predictions at 50..800 ms from execution start."""
    require(startup.phase in ("ready", "holding") and startup.hold is not None,
            "A full chunk can start only while holding after initialization")
    feedback.fresh(now_ns)
    require(prediction.get("epoch") == startup.epoch, "Prediction belongs to another initialization session")
    profile = startup.profile
    expected = profile["policy"]["joint_names"]
    names = prediction.get("joint_names")
    if profile["robot"] == SG2:
        require(isinstance(names, list) and len(names) == len(expected) and
                all(isinstance(n, str) for n in names) and set(names) == set(expected),
                "Expected each SG2 arm and gripper joint exactly once")
    else:
        require(names == expected, "Prediction joint order mismatch")
    require(prediction.get("units") == ["rad"] * len(expected), "Expected recorded radian joint coordinates")
    steps = profile["policy"]["future_steps"]
    interval = profile["policy"]["frame_interval_ns"]
    positions = prediction["positions"]
    if profile["robot"] == SG2:
        require(isinstance(positions, list) and all(isinstance(row, list) and
                all(type(value) in (int, float) for value in row) for row in positions),
                "SG2 positions must be JSON numeric arrays")
    values = np.asarray(positions, dtype=np.float64)
    safety_require(values.shape == (steps, len(expected)) and np.isfinite(values).all(),
                   f"Expected finite {steps}-by-{len(expected)} predictions")
    source = prediction.get("source_time_ns")
    require(type(source) is int and 0 < source <= now_ns, "Invalid/future observation timestamp")
    offsets = [(i + 1) * interval for i in range(steps)]
    result = {}
    for group in command_groups(profile):
        spec = profile["commands"][group]
        joints = spec["joint_names"]
        if group not in policy_groups(profile):
            rows = [[startup.hold[n] for n in joints] for _ in offsets]
        else:
            rows = values[:, [names.index(n) for n in joints]]
        if profile["robot"] == SG2:
            for column, joint in enumerate(joints):
                limit = profile["reference_limits"][joint]
                targets = rows[:, column]
                safety_require(((limit["lower"] <= targets) & (targets <= limit["upper"])).all(),
                               f"SG2 target outside position limits: {joint}")
                changes = np.diff(targets, prepend=feedback.positions[joint])
                safety_require((np.abs(changes) <= limit["velocity"] * interval / 1e9 + 1e-9).all(),
                               f"SG2 target exceeds joint velocity limit: {joint}")
        check_positions(joints, rows[0], profile)
        result[group] = {"topic": spec["topic"], "message": message(joints, rows, offsets, now_ns)}
    return result


def hold_messages(feedback, startup, now_ns, *, include_right=False, right_positions=None, stop_all=False):
    """Hold active policy arms at the commanded target, or all arms at fresh measurements.

    The include_right and right_positions keywords also cover the left arm on SG2.
    """
    feedback.fresh(now_ns)
    require(startup.hold is not None, "Hold pose was not captured")
    result = {}
    for group in command_groups(startup.profile):
        active = group in policy_groups(startup.profile)
        if active and not include_right:
            continue
        spec = startup.profile["commands"][group]
        if stop_all:
            row = [feedback.positions[n] for n in spec["joint_names"]]
        elif active:
            target = startup.hold if right_positions is None else right_positions
            row = [target[n] for n in spec["joint_names"]]
        else:
            row = [startup.hold[n] for n in spec["joint_names"]]
        check_positions(spec["joint_names"], row, startup.profile)
        result[group] = {"topic": spec["topic"], "message": message(spec["joint_names"], [row], [100_000_000], now_ns)}
    return result
