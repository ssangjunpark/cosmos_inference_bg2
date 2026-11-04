"""Synchronous full chunks, measured-position pause/resume, and terminal faults."""
from collections import deque
import re
import uuid

from .core import (MAX_ANGULAR_ERROR, MAX_ANGULAR_VELOCITY, angular_names,
                   hold_messages, require, safety_require, trajectories)

SCHEMA = "cosmos_bg2_joint_chunk_v1"  # Model RPC remains output-only and unchanged.
EXECUTION_CONTRACT = "synchronous_full_chunk_v2"
STATUS_SCHEMA = "cosmos_bg2_status_v3"


class ChunkExecutor:
    """Execute all points, publish the final-command hold, then open the next cycle."""
    def __init__(self, startup):
        self.startup = startup
        self.cycle, self.ready_since_ns, self.execution_end_ns = 0, 0, 0
        self.last_source_ns = 0
        self.last_commands, self.last_request_id, self.last_completed_request_id = None, None, None
        self.last_clock, self.start_positions, self.right_hold = None, None, None
        self.seen = deque(maxlen=128)

    def clock(self, robot_ns, monotonic_ns):
        if self.last_clock is not None:
            robot_before, mono_before = self.last_clock
            require(monotonic_ns >= mono_before, "Monotonic clock moved backwards")
            require(abs((robot_ns - robot_before) - (monotonic_ns - mono_before)) <= 100_000_000,
                    "Robot clock jumped; restart the session")
        self.last_clock = robot_ns, monotonic_ns

    def accept(self, chunk, feedback, now_ns, *, submit=None):
        require(isinstance(chunk, dict) and chunk.get("schema") == SCHEMA, "Unknown chunk schema")
        require(chunk.get("execution_contract") == EXECUTION_CONTRACT, "Synchronous execution contract required")
        require(chunk.get("epoch") == self.startup.epoch, "Prediction belongs to another initialization session")
        ident = chunk.get("request_id")
        require(isinstance(ident, str) and re.fullmatch(r"[0-9a-f]{32}", ident), "Invalid request ID")
        require(ident not in self.seen, "Duplicate chunk")
        require(self.startup.phase in ("ready", "holding") and self.ready_since_ns > 0,
                "Current chunk must finish and hold before another can start")
        require(type(chunk.get("cycle")) is int and chunk["cycle"] == self.cycle, "Wrong execution cycle")
        source = chunk.get("source_time_ns")
        require(type(source) is int and source >= self.ready_since_ns and source > self.last_source_ns,
                "Observation must be captured after the preceding hold")
        commands = trajectories(chunk, feedback, self.startup, now_ns, held_command=self.right_hold)
        if submit is not None:
            submit(commands)
        self.start_positions = dict(self.startup.hold)
        self.start_positions.update(self.right_hold)
        self.last_source_ns, self.last_request_id = source, ident
        self.last_commands = commands
        self.execution_end_ns = now_ns + 800_000_000
        self.right_hold = dict(zip(chunk["joint_names"], chunk["positions"][-1]))
        self.seen.append(ident)
        self.startup.phase = "running"
        return commands

    def expected_right(self, source_ns):
        if self.startup.phase != "running" or self.last_commands is None:
            return self.right_hold or self.startup.hold
        msg = self.last_commands["arm_right"]["message"]
        start = msg["header"]["stamp"]["sec"] * 10**9 + msg["header"]["stamp"]["nanosec"]
        elapsed = max(0, source_ns - start)
        before, previous = 0, [self.start_positions[n] for n in msg["joint_names"]]
        for point in msg["points"]:
            after = point["time_from_start"]["sec"] * 10**9 + point["time_from_start"]["nanosec"]
            if elapsed <= after:
                alpha = max(0., min(1., (elapsed - before) / (after - before)))
                return dict(zip(msg["joint_names"], [a + alpha * (b - a) for a, b in zip(previous, point["positions"])]))
            before, previous = after, point["positions"]
        return dict(zip(msg["joint_names"], previous))

    def monitor(self, feedback, now_ns):
        feedback.fresh(now_ns)
        if self.startup.phase not in ("ready", "running", "holding", "paused"):
            return
        expected = dict(self.startup.hold)
        expected.update(self.expected_right(feedback.source_ns))
        for name in angular_names(self.startup.profile):
            safety_require(abs(feedback.velocities[name]) <= MAX_ANGULAR_VELOCITY,
                           f"Measured angular velocity exceeds 2 rad/s: {name}")
            safety_require(abs(feedback.positions[name] - expected[name]) <= MAX_ANGULAR_ERROR + 1e-9,
                           f"Angular tracking error exceeds 10 degrees: {name}")

    def tick(self, feedback, now_ns, *, submit=None, monitor=True):
        if self.startup.phase not in ("ready", "running", "holding", "paused"):
            return None
        feedback.fresh(now_ns)
        if monitor:
            self.monitor(feedback, now_ns)
        if self.startup.phase == "ready" and not self.ready_since_ns:
            commands = hold_messages(feedback, self.startup, now_ns, include_right=True)
            if submit is not None:
                submit(commands)
            self.right_hold = {n: self.startup.hold[n] for n in self.startup.profile["policy"]["joint_names"]}
            self.ready_since_ns = now_ns
            return commands
        if self.startup.phase == "running" and now_ns >= self.execution_end_ns:
            commands = hold_messages(feedback, self.startup, now_ns, include_right=True, right_positions=self.right_hold)
            if submit is not None:
                submit(commands)
            self.startup.phase = "holding"
            self.execution_end_ns = 0
            self.last_completed_request_id = self.last_request_id
            self.cycle += 1
            self.ready_since_ns = now_ns
            return commands
        return None

    def pause(self, feedback, now_ns, *, submit=None, monitor=True):
        require(self.startup.phase in ("ready", "running", "holding") and self.ready_since_ns > 0,
                "Only an initialized active session can pause")
        if monitor:
            self.monitor(feedback, now_ns)
        commands = hold_messages(feedback, self.startup, now_ns, include_right=True, stop_all=True)
        if submit is not None:
            submit(commands)
        self.startup.hold = dict(feedback.positions)
        self.right_hold = {n: feedback.positions[n] for n in self.startup.profile["policy"]["joint_names"]}
        self.execution_end_ns, self.ready_since_ns = 0, 0
        self.last_commands = commands
        self.startup.phase, self.startup.epoch = "paused", uuid.uuid4().hex
        self.cycle += 1
        return commands

    def resume(self, feedback, now_ns, *, submit=None, monitor=True):
        require(self.startup.phase == "paused", "Only a paused session can resume; faults remain terminal")
        if monitor:
            self.monitor(feedback, now_ns)
        commands = hold_messages(feedback, self.startup, now_ns, include_right=True, stop_all=True)
        if submit is not None:
            submit(commands)
        self.startup.hold = dict(feedback.positions)
        self.right_hold = {n: feedback.positions[n] for n in self.startup.profile["policy"]["joint_names"]}
        self.startup.phase, self.startup.epoch = "holding", uuid.uuid4().hex
        self.cycle += 1
        self.execution_end_ns, self.ready_since_ns = 0, now_ns
        self.last_commands, self.last_request_id, self.last_completed_request_id = commands, None, None
        return commands

    def stop(self, feedback, now_ns, *, fault=False):
        self.execution_end_ns, self.ready_since_ns = 0, 0
        self.startup.phase = "fault" if fault else "stopped"
        if self.startup.hold is None or feedback is None:
            return None
        return hold_messages(feedback, self.startup, now_ns, include_right=True, stop_all=True)

    def status(self, now_ns):
        return {"schema": STATUS_SCHEMA, "execution_contract": EXECUTION_CONTRACT,
                "epoch": self.startup.epoch, "phase": self.startup.phase, "robot_now_ns": now_ns,
                "cycle": self.cycle, "ready_since_ns": self.ready_since_ns,
                "execution_end_ns": self.execution_end_ns,
                "remaining_ns": max(0, self.execution_end_ns - now_ns),
                "last_request_id": self.last_request_id, "last_completed_request_id": self.last_completed_request_id,
                "last_source_time_ns": self.last_source_ns,
                "needs_chunk": self.ready_since_ns > 0 and self.startup.phase in ("ready", "holding")}
