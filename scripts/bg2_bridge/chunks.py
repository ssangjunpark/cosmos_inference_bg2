"""Robot-side chunk lifecycle; the existing JointTrajectoryController executes the points."""
from collections import deque
from dataclasses import dataclass
import re

from .core import BridgeError, hold_messages, require, trajectories

SCHEMA = "cosmos_bg2_joint_chunk_v1"


class ChunkExecutor:
    """One current chunk, no inference FIFO, and immutable original forecast times."""
    def __init__(self, startup, *, velocity_scale=.25, tracking_tolerance=.1):
        self.startup = startup
        self.velocity_scale, self.tracking_tolerance = velocity_scale, tracking_tolerance
        self.last_source_ns, self.valid_until_ns = 0, 0
        self.seen = deque(maxlen=128)
        self.last_commands, self.last_request_id = None, None
        self.last_clock = None
        self.right_hold = None

    def clock(self, robot_ns, monotonic_ns):
        if self.last_clock is not None:
            before_robot, before_monotonic = self.last_clock
            require(monotonic_ns >= before_monotonic, "Monotonic clock moved backwards")
            require(abs((robot_ns - before_robot) - (monotonic_ns - before_monotonic)) <= 100_000_000,
                    "Robot clock jumped; restart the session")
        self.last_clock = robot_ns, monotonic_ns

    def accept(self, chunk, feedback, now_ns, *, submit=None):
        require(isinstance(chunk, dict), "Chunk must be a JSON object")
        require(chunk.get("schema") == SCHEMA, "Unknown chunk schema")
        identifier = chunk.get("request_id")
        require(isinstance(identifier, str) and re.fullmatch(r"[0-9a-f]{32}", identifier), "Invalid request ID")
        require(identifier not in self.seen, "Duplicate chunk")
        require(type(chunk.get("source_time_ns")) is int and chunk["source_time_ns"] > self.last_source_ns,
                "Old/out-of-order observation")
        # Validate everything before replacing the current chunk. A malformed reply
        # leaves the last accepted trajectory and its existing deadline untouched.
        commands = trajectories(chunk, feedback, self.startup, now_ns,
                                velocity_scale=self.velocity_scale, tracking_tolerance=self.tracking_tolerance)
        if submit is not None:
            submit(commands)
        self.last_source_ns = chunk["source_time_ns"]
        self.valid_until_ns = self.last_source_ns + 800_000_000
        self.last_commands, self.last_request_id = commands, identifier
        self.seen.append(identifier)
        self.startup.phase = "running"
        self.right_hold = None
        return commands

    def check_tracking(self, feedback):
        if self.last_commands is None:
            return
        msg = self.last_commands["arm_right"]["message"]
        start = msg["header"]["stamp"]["sec"] * 1_000_000_000 + msg["header"]["stamp"]["nanosec"]
        points = msg["points"]
        elapsed = max(0, feedback.source_ns - start)
        previous = points[0]
        expected = previous["positions"]
        for point in points[1:]:
            before = previous["time_from_start"]["sec"] * 1_000_000_000 + previous["time_from_start"]["nanosec"]
            after = point["time_from_start"]["sec"] * 1_000_000_000 + point["time_from_start"]["nanosec"]
            if elapsed <= after:
                alpha = max(0., min(1., (elapsed - before) / (after - before)))
                expected = [a + alpha * (b - a) for a, b in zip(previous["positions"], point["positions"])]
                break
            previous, expected = point, point["positions"]
        require(all(abs(feedback.positions[name] - target) <= self.tracking_tolerance
                    for name, target in zip(msg["joint_names"], expected)), "Right-arm tracking diverged")

    def tick(self, feedback, now_ns, *, check_tracking=True):
        if self.startup.phase not in ("ready", "running", "holding"):
            return None
        feedback.fresh(now_ns)
        require(all(abs(feedback.positions[name] - self.startup.hold[name]) <= self.tracking_tolerance
                    for group in self.startup.profile["hold_groups"]
                    for name in self.startup.profile["commands"][group]["joint_names"]), "Held group drifted")
        if check_tracking and self.startup.phase in ("ready", "holding"):
            right_reference = self.startup.hold if self.startup.phase == "ready" else self.right_hold
            if right_reference is not None:
                require(all(abs(feedback.positions[n] - right_reference[n]) <= self.tracking_tolerance
                            for n in self.startup.profile["policy"]["joint_names"]), "Right-arm hold drifted")
        if self.startup.phase == "running":
            if check_tracking:
                self.check_tracking(feedback)
            if now_ns >= self.valid_until_ns:
                self.startup.phase = "holding"
                self.valid_until_ns = 0
                self.right_hold = {n: feedback.positions[n] for n in self.startup.profile["policy"]["joint_names"]}
                return hold_messages(feedback, self.startup, now_ns, include_right=True)
        return None

    def stop(self, feedback, now_ns, *, fault=False):
        self.valid_until_ns = 0
        if fault:
            self.startup.phase = "fault"
        else:
            self.startup.phase = "holding"
        if self.startup.hold is None or feedback is None:
            return None
        self.right_hold = {n: feedback.positions[n] for n in self.startup.profile["policy"]["joint_names"]}
        return hold_messages(feedback, self.startup, now_ns, include_right=True)

    def status(self, now_ns):
        return {"schema": "cosmos_bg2_status_v1", "epoch": self.startup.epoch,
                "phase": self.startup.phase, "robot_now_ns": now_ns,
                "last_request_id": self.last_request_id, "last_source_time_ns": self.last_source_ns,
                "valid_until_ns": self.valid_until_ns,
                "remaining_ns": max(0, self.valid_until_ns - now_ns),
                "needs_chunk": self.startup.phase in ("ready", "holding")}


@dataclass
class RequestPlanner:
    """5090-side prefetch decision, using a local monotonic clock and FFW status time."""
    latency_budget_ns: int
    margin_ns: int = 100_000_000
    in_flight: bool = False
    last_source_ns: int = 0

    def should_request(self, status, received_monotonic_ns, now_monotonic_ns, observation_source_ns):
        require(self.latency_budget_ns > 0 and self.margin_ns >= 0, "Invalid measured latency budget")
        age = now_monotonic_ns - received_monotonic_ns
        if self.in_flight or not 0 <= age <= 150_000_000 or status["phase"] not in ("ready", "running", "holding"):
            return False
        if observation_source_ns <= self.last_source_ns:
            return False
        remaining = max(0, status["remaining_ns"] - age)
        return remaining <= self.latency_budget_ns + self.margin_ns

    def started(self, observation_source_ns):
        require(not self.in_flight and observation_source_ns > self.last_source_ns, "Inference is already in flight or observation repeated")
        self.in_flight, self.last_source_ns = True, observation_source_ns

    def finished(self, elapsed_ns):
        require(elapsed_ns > 0, "Invalid measured request latency")
        self.in_flight = False
        # Conservative envelope for this session; reset/reconfigure after an explicit
        # warm-up rather than letting a cold request define a motion cadence.
        self.latency_budget_ns = max(self.latency_budget_ns, elapsed_ns)
