# Synchronous Cosmos execution contract

Agreed with the user on 2026-11-04. This supersedes the earlier asynchronous bridge design.
Implementation and bundles are prepared on PRO 6000. Updating this source does not deploy
it to 5090 or FFW, and no motion was started during this change.

## Current startup and pause/resume behavior (2026-11-05)

The [normal execution guide](policy_execution.md) supersedes the earlier mandatory
fresh-homing restart procedure below. Current-pose startup is the default, with the
existing 500 ms settling check; fresh homing is optional (`COSMOS_INITIALIZATION=homing`).
Normal gateway Ctrl+C pauses at measured positions, confirms the paused epoch and closes
recordings. A new gateway prompt resumes from fresh state/images without restarting the
companion or homing. Faults remain terminal. Status schema is `cosmos_bg2_status_v3` and
execution contract `synchronous_full_chunk_v2`; update both deployment copies once.
Negative predicted right-gripper targets are floored to 0 on PRO 6000 as authorized;
the bridge forwards the resulting 16 targets unchanged. Historical startup/validation
notes below describe the earlier revision, not the current default.

## One cycle

1. ROBOTIS completes the session's initial homing. The companion observes fresh success
   for both arms, head and elevator, checks 500 ms settling, and captures the measured pose.
2. FFW establishes its initial hold and publishes a ready session/cycle with `ready_since_ns`.
3. The 5090 waits for joint feedback and all three camera frames captured at or after that
   hold timestamp. The policy receives the newly measured right-arm/gripper state, fresh
   views and the supplied instruction. It does not receive the last prediction as its state.
4. A blocking inference request runs on PRO 6000 while FFW holds. There is no inference
   prefetch, overlapping motion, action-prefix conditioning, or queue of observations.
5. FFW validates the response and submits exactly 16 unchanged target rows, at 50, 100,
   ..., 800 ms from execution start. There is no extra 20 ms delay, measured-position
   waypoint, dropped prefix, speed scaling, clipping or gripper remapping.
6. The existing ROBOTIS controllers interpolate the whole trajectory. Until its scheduled
   end the gateway cannot request another chunk and FFW cannot replace its active chunk.
7. At the scheduled end FFW publishes a hold using the last commanded right-arm/gripper
   target. Left arm/gripper, head and elevator keep their captured session targets.
8. Only after the hold publication succeeds does FFW increment its cycle and acknowledge
   `last_completed_request_id`. The 5090 then collects a new post-hold observation.

The 0.8 seconds is execution time, independent of inference duration. Observation timestamps
remain provenance and post-hold freshness checks; they are not motion deadlines. A one-second
inference does not discard any of the 16 points. The next cycle adds another inference pause.

Completion here is the scheduled end of the submitted trajectory and successful publication
of the final hold. It is not a FollowJointTrajectory action-result acknowledgement or proof
of exact measured arrival. Tracking is monitored separately with the operator's limit below.
A lost chunk/acknowledgement leaves the gateway waiting and FFW holding; it does not start
another request from a stale ready message. Restart after diagnosing the delivery failure.

## Operator-specified monitoring (updated 2026-11-08)

- Command-step velocity: no target-step rejection, scaling or retiming. The operator
  selected measured velocity only.
- Measured motion: while execute mode is initialized, stop if any monitored angular joint
  reports absolute velocity above **9999 rad/s**.
- Tracking: print a warning if any angular joint differs by more than **10 degrees** from its current
  command reference; do not stop, reject or alter the chunk. Warnings include the measured/reference
  positions and error in degrees, at most once per second per joint while above the threshold.
  The right-arm reference interpolates the commanded waypoints from fresh measured state at acceptance; passive
  angular joints use captured holds. Holding uses the last commanded right target.
- These are absolute thresholds, not multipliers. No predicted target is scaled or retimed.
- The elevator is prismatic. Neither rad/s nor degrees is applied to it. Its captured hold
  and feedback/controller checks remain; no new linear threshold is guessed.
- Nonfinite predicted targets raise a terminal safety fault. Bridge position-range rejection
  has been removed for all joints; the ROBOTIS controller/driver configuration is unchanged.

A motion-limit fault sends a best-effort hold at fresh measured positions for all groups,
latches `fault`, clears request readiness, and prevents automatic continuation. If feedback
or command ownership is unavailable, publishing that hold may be impossible. This is a
software stop, not torque disable or a hardware emergency stop. Physical stopping and
tracking have not been commissioned.

The old `velocity_scale=0.25` and generic `tracking_tolerance=0.1` are removed. The latter
no longer applies a shared angular/linear tolerance to the elevator. Only the source-matched
training normalization/inverse normalization remains in the model path.

## Startup and restarting

Item 13 is retained: each new companion process/session must witness fresh, externally
managed ROBOTIS homing before becoming ready. The companion does not start homing itself.
It does not home between chunks. Restarting the PRO 6000 model server alone does not restart
the companion. An explicit gateway stop after execution closes the companion session;
resuming after a stop/fault requires restarting the companion before fresh homing.

The selected SNUPI YAML remains unchanged. Its head target 0.78 conflicts with the installed
Rev4 upper limit 0.6951, and the existing home validation still rejects it. Camera transforms
and the reference profile remain unverified/disabled. These are existing commissioning
blockers; this implementation does not bypass them.

## Code and protocol

- `scripts/bg2_bridge/gateway.py`: blocking request, post-hold observations, wait for the
  matching FFW cycle/completed request before another request.
- `scripts/bg2_bridge/chunks.py`: initial hold, full-chunk lifecycle, final-command hold,
  terminal stop, operator-specified angular monitoring.
- `scripts/bg2_bridge/core.py`: unchanged-target trajectory construction and limit checks.
- `deploy/ffw/cosmos_ffw/node.py`: native ROS subscriptions, controller gates, fault-to-stop.
- `scripts/robotis_bridge.py`: offline replay only; the old live async loop is removed.

The model's HTTP prediction schema stays `cosmos_bg2_joint_chunk_v1`; model loading and
normalization are unchanged. The gateway adds `execution_contract=synchronous_full_chunk_v1`
and the current integer `cycle` to forwarded chunks. FFW requires both fields.
Status uses `cosmos_bg2_status_v2` with the same contract, `cycle`, `ready_since_ns`,
`execution_end_ns`, `last_request_id`, `last_completed_request_id`, and `needs_chunk`.
An old async gateway/companion cannot satisfy the new handshake. Update both deployment
folders together. The model server needs no restart for this bridge-only protocol change.

## Validation

The local CPU suite runs 90 tests: 89 pass and one optional SDK check skips in the host
environment; that SDK check also passes separately in the isolated local SDK environment. Coverage includes a real local HTTP request held longer than 0.8 seconds,
all 16 targets unchanged, delayed/stale status, no mid-chunk request or replacement,
post-hold camera/joint timestamps, final commanded holds despite different measurements,
9999 rad/s stops and warning-only 10 degree boundaries, terminal ROS-node stop behavior, old-protocol rejection,
feedback/ownership faults, and bundle portability. This is software validation only.
