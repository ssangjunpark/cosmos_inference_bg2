# Remaining runtime checks — 2026-11-05 KST

This is an audit of current code, not a change to the thresholds below.

| Check | Current value/behavior | Source/purpose |
| --- | --- | --- |
| Planned angular velocity | 2 rad/s, including the first 50 ms segment | Operator requested |
| Measured angular velocity | 2 rad/s, one feedback sample can fault | Operator requested |
| Angular tracking error | 10 degrees against interpolated commanded reference | Operator requested |
| Gripper floor | Negative predicted right-gripper targets become zero | Operator requested; no other position clipping |
| Initial capture | 500 ms continuous fresh, stable measured feedback | Bridge startup assumption |
| Initial position stability | YAML position_tolerance, currently 0.1 rad for angular joints and 0.1 m for lift | Existing YAML reused by bridge |
| Current-pose startup velocity | 2 rad/s for angular joints; lift retains YAML 0.01 m/s | Angular startup gate corrected from 0.01 after measured gripper jitter |
| Optional homing-mode settling | YAML velocity_tolerance, currently 0.01; fresh successful homing statuses required | Only when initialization=homing; current mode is default |
| Joint feedback freshness | At most 100 ms source/receipt age; at most 25 ms future timestamp | Bridge timing assumption |
| Camera freshness | At most 100 ms source/receipt age; at most 25 ms future timestamp | Bridge timing assumption |
| Camera/joint alignment | Nearest camera header within 25 ms of selected joint state | Matches supplied training preprocessing |
| FFW status freshness | 150 ms | Gateway waits when missing/stale |
| Robot-clock discontinuity | More than 100 ms difference from monotonic-clock progress faults | Bridge clock assumption |
| Controller polling | Every 250 ms, query timeout 750 ms, health lease 1 second | Bridge controller assumptions |
| Command endpoint discovery | 1 second at activation | Bridge ROS discovery assumption |
| Publisher ownership | One own publisher; any foreign registration blocks/faults, even if idle | Bridge exclusivity assumption, not an encoder tolerance |
| Controller subscriptions | At least one subscriber on every command topic | Bridge graph check |
| Controller interfaces | Four named controllers active and claiming expected position interfaces | Installed deployment contract |
| Shutdown control acknowledgement | Up to 1 second; absence is reported, not treated as success | Bridge pause/stop handshake |
| RPC | 10-second default timeout; allowed maximum 120 seconds | Client timeout, not trajectory duration |
| Retry pacing | 100 ms for failed inference and unacknowledged resume | No motion prefetch or parallel policy calls |

Additional format/integrity checks remain: finite positions/velocities, correct joint
names/order/count, all three verified camera transforms/dimensions, 16×8 targets in
radians with 50 ms spacing, increasing timestamps, fresh post-hold/resume inputs,
matching schema/session epoch/cycle/request ID, no duplicate or overlapping chunks,
and completion acknowledgement before the next request. `in_flight` is a boolean around
a blocking inference call; it does not enable asynchronous motion or RTC.

Payload caps: 16 MiB per compressed camera frame, 8 MiB observation NPZ, 128 KiB
prediction/chunk/status envelopes, 1 KiB control messages, and 2048 instruction
characters. Native decoder checks the conditioned current row within 2e-5; this is a
model-output integrity check, not a requirement for an actuator to reach a target.
Training normalization retains its original 1e-6 denominator floor.

Bridge position-range rejection for all joints and the former quarter-speed gate are
removed. ROBOTIS controller/driver settings were not modified by those changes.

## SmolVLA publisher handoff

The registered right-arm publisher was `smolvla_main_output`, in the same Zenoh session
as `lerobot_main_process` and `lerobot_engine_client`. It came from `main-runtime` inside
5090's existing `lerobot_server`. No right-arm command messages were observed during a
2-second passive sample; registration alone caused the ownership fault.

With explicit user approval, only that service was stopped:

```bash
# On 5090
docker exec lerobot_server /command/s6-svc -d /run/service/main-runtime
```

Verified afterwards: `smolvla_main_output` registrations were gone, the SmolVLA engine
process retained the same PID, and the ROBOTIS follower/cameras were left running.
Cosmos was not restarted automatically. Its fault is latched, so stop the 5090 gateway,
restart the FFW companion, then rerun the gateway; no homing is required.

To restore the prior client later, after stopping Cosmos execution:

```bash
# On 5090
docker exec lerobot_server /command/s6-svc -u /run/service/main-runtime
```

Source: `scripts/bg2_bridge/{core,chunks,observations,gateway,rpc}.py`,
`scripts/gateway.py`, `scripts/bg2_inference.py`, and `deploy/ffw/cosmos_ffw/node.py`.
