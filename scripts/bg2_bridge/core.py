"""Transport-independent joint feedback, homing, and timestamped trajectory construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import uuid

import numpy as np

GROUPS = ("arm_right", "arm_left", "head", "lift")
CONTROLLERS = {"arm_right": "arm_r_controller", "arm_left": "arm_l_controller",
               "head": "head_controller", "lift": "lift_controller"}


class BridgeError(ValueError):
    pass


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
    return [name for group in GROUPS for name in profile["commands"][group]["joint_names"]]


def validate_profile(profile, *, live=False, execute=False):
    require(profile.get("schema") == 1 and profile.get("robot") == "ffw_bg2_rev4", "Unsupported robot profile")
    names = all_names(profile)
    require(len(names) == len(set(names)) == 19, "Expected 19 unique commanded joints")
    expected = [f"arm_r_joint{i}" for i in range(1, 8)] + ["gripper_r_joint1"]
    require(profile["commands"]["arm_right"]["joint_names"] == profile["policy"]["joint_names"] == expected,
            "Right-arm joint order must match the checkpoint")
    require(profile["policy"]["future_steps"] == 16 and profile["policy"]["frame_interval_ns"] == 50_000_000,
            "Expected the trained 16-step, 20 Hz horizon")
    for name in names:
        limit = profile["reference_limits"][name]
        require(all(math.isfinite(limit[k]) for k in ("lower", "upper", "velocity")) and
                limit["lower"] < limit["upper"] and limit["velocity"] > 0, f"Invalid limits for {name}")
    for group in GROUPS:
        command = profile["commands"][group]
        require(command["msg_type"] == "trajectory_msgs/msg/JointTrajectory" and command["topic"].startswith("/"),
                "Expected an absolute JointTrajectory topic")
    require(len({profile["commands"][g]["topic"] for g in GROUPS}) == 4, "Command topics must be distinct")
    require(set(profile["cameras"]) == {"head", "wrist", "wrist_left"}, "Three camera views are required")
    if live:
        for key, camera in profile["cameras"].items():
            require(camera.get("live_to_training_rotation_deg") in (0, 90, 180, 270), f"{key}: verify camera rotation first")
            wh = camera.get("raw_wh")
            require(camera.get("raw_dimensions_verified") is True and isinstance(wh, list) and len(wh) == 2 and
                    all(isinstance(v, int) and v > 0 for v in wh), f"{key}: verify raw_wh first")
    if execute:
        require(profile.get("execution_enabled") is True and profile.get("status") == "verified_on_robot",
                "Execution requires a verified deployment profile; reference profiles cannot command hardware")


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
    for group, controller in CONTROLLERS.items():
        key = controller.replace("_controller", "_joint_trajectory_executor")
        data = executors[key]["ros__parameters"]
        names = data["joint_names"]
        require(len(names) == len(set(names)) and set(names) == set(profile["commands"][group]["joint_names"]), f"Homing joint mismatch: {group}")
        positions = data[data["step_names"][-1]]
        require(len(names) == len(positions), "Homing pose shape mismatch")
        require(data["action_topic"] == f"/{controller}/follow_joint_trajectory", "Invalid homing action")
        pose = dict(zip(names, map(float, positions)))
        for name, value in pose.items():
            limit = profile["reference_limits"][name]
            require(math.isfinite(value) and limit["lower"] <= value <= limit["upper"], f"Homing pose outside limits: {name}")
        targets[group] = {"positions": pose, "position_tolerance": float(data["position_tolerance"]),
                          "velocity_tolerance": float(data["velocity_tolerance"]),
                          "status_topic": data["action_topic"] + "/_action/status"}
        require(all(math.isfinite(targets[group][k]) and targets[group][k] >= 0 for k in ("position_tolerance", "velocity_tolerance")),
                "Invalid homing tolerances")
        # ROS action transport uses /<action-name>/_action/status.
    return targets


class Startup:
    def __init__(self, profile, home, started_ns, settle_ns=500_000_000):
        self.profile, self.home, self.started_ns, self.settle_ns = profile, home, started_ns, settle_ns
        self.phase, self.epoch = "initializing", uuid.uuid4().hex
        self.completed, self.hold = set(), None
        self.goal_starts = {}
        self.stable_since = self.last_feedback_ns = None

    def homing_status(self, group, status, goal_started_ns):
        require(group in GROUPS, "Unknown homing group")
        if goal_started_ns < self.started_ns:
            return
        if self.phase in ("ready", "running", "holding"):
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
            if self.completed == set(GROUPS):
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
        settled = all(abs(feedback.positions[n] - target) <= data["position_tolerance"] and
                      abs(feedback.velocities[n]) <= data["velocity_tolerance"]
                      for data in self.home.values() for n, target in data["positions"].items())
        if not settled:
            self.stable_since = None
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
    for name, value in zip(names, values):
        limit = profile["reference_limits"][name]
        require(math.isfinite(float(value)) and limit["lower"] <= value <= limit["upper"], f"Target outside limits: {name}")


def message(names, rows, offsets, start_ns):
    return {"header": {"stamp": ros_time(start_ns), "frame_id": ""}, "joint_names": list(names),
            "points": [{"positions": list(map(float, row)), "velocities": [], "accelerations": [], "effort": [],
                        "time_from_start": ros_time(offset)} for row, offset in zip(rows, offsets)]}


def trajectories(prediction, feedback, startup, now_ns, *, lead_ns=20_000_000, velocity_scale=0.25,
                 tracking_tolerance=0.1, prefix=16):
    require(startup.phase in ("ready", "running", "holding") and startup.hold is not None, "Initialization and hold capture are incomplete")
    feedback.fresh(now_ns)
    require(prediction.get("epoch") == startup.epoch, "Prediction belongs to another initialization session")
    require(prediction.get("joint_names") == startup.profile["policy"]["joint_names"], "Prediction joint order mismatch")
    require(prediction.get("units") == ["rad"] * 8, "Expected recorded radian joint coordinates")
    values = np.asarray(prediction["positions"], dtype=np.float64)
    require(values.shape == (16, 8) and np.isfinite(values).all(), "Expected finite 16-by-8 predictions")
    require(0 < velocity_scale <= 1 and 1 <= prefix <= 16 and lead_ns >= 0, "Invalid trajectory settings")
    source = int(prediction["source_time_ns"])
    require(0 <= now_ns - source, "Prediction is from the future; check clocks")
    start = now_ns + lead_ns
    future = [(i, source + (i + 1) * 50_000_000 - start) for i in range(16)
              if source + (i + 1) * 50_000_000 - start >= 50_000_000][:prefix]
    require(bool(future), "Prediction horizon expired; do not retimestamp old predictions")
    right = startup.profile["commands"]["arm_right"]["joint_names"]
    for row in values:
        check_positions(right, row, startup.profile)
    offsets = [0] + [offset for _, offset in future]
    right_rows = [[feedback.positions[n] for n in right]] + [values[i].tolist() for i, _ in future]
    check_positions(right, right_rows[0], startup.profile)
    for previous, row, before, after in zip(right_rows, right_rows[1:], offsets, offsets[1:]):
        speeds = np.abs(np.subtract(row, previous)) / ((after - before) / 1e9)
        require(all(speed <= startup.profile["reference_limits"][n]["velocity"] * velocity_scale for n, speed in zip(right, speeds)),
                "Predicted motion exceeds configured velocity limits")
    result = {}
    for group in GROUPS:
        spec = startup.profile["commands"][group]
        names = spec["joint_names"]
        if group == "arm_right":
            rows = right_rows
        else:
            captured = [startup.hold[n] for n in names]
            require(all(abs(feedback.positions[n] - startup.hold[n]) <= tracking_tolerance for n in names), "Held group drifted")
            check_positions(names, captured, startup.profile)
            rows = [captured for _ in offsets]
        result[group] = {"topic": spec["topic"], "message": message(names, rows, offsets, start)}
    return result


def hold_messages(feedback, startup, now_ns, *, include_right=False):
    feedback.fresh(now_ns)
    require(startup.hold is not None, "Hold pose was not captured")
    result = {}
    for group in GROUPS:
        if group == "arm_right" and not include_right:
            continue
        spec = startup.profile["commands"][group]
        row = [feedback.positions[n] if group == "arm_right" else startup.hold[n] for n in spec["joint_names"]]
        check_positions(spec["joint_names"], row, startup.profile)
        result[group] = {"topic": spec["topic"], "message": message(spec["joint_names"], [row], [100_000_000], now_ns)}
    return result
