# Cosmos FFW companion bundle

This companion runs beside the existing ROBOTIS bringup. The model stays on PRO 6000;
5090 handles observations and blocking inference requests. FFW uses native ROS to command
its existing position JointTrajectoryControllers. No new motor driver or IK is included.

## Prepare and transfer

On PRO 6000, run `python3 scripts/package_ffw.py --output transfer/ffw` with a new output
folder. The builder copies shared modules, source templates, disabled reference profile
and a checksum manifest. It never contacts either remote machine. Transfer only when
requested. `deploy/ffw` is source; `transfer/ffw` is the complete generated folder.

## Synchronous behavior

1. Start from current pose (default): check fresh settled measured feedback for 500 ms,
   capture all holds, then publish readiness. Optional `COSMOS_INITIALIZATION=homing`
   requires fresh ROBOTIS homing instead.
2. 5090 collects fresh measured joints and all three cameras after the hold timestamp.
   PRO 6000 calculates one prediction while FFW holds.
3. FFW sends all 16 unchanged right-arm/gripper targets at 50..800 ms from execution start.
   The existing controllers interpolate them. Passive groups keep their captured targets.
4. No request or active-chunk replacement occurs during execution. At the scheduled end,
   FFW explicitly holds the last commanded target and acknowledges the completed cycle.
5. Only then does 5090 collect a new observation and request the next chunk.

No prefetch planner, observation-based execution expiry, skipped prefix, extra 20 ms lead,
measured-position waypoint or velocity multiplier remains. Model normalization/inversion
matches training. Full details are in `docs/synchronous_execution_20261104.md` in the repo.

## Stop limits

Angular planned/measured velocity above **2 rad/s** or tracking error above **10 degrees**
causes a terminal session fault and a best-effort fresh measured-position stop hold.
The angle limits apply to angular joints, including the grippers and head, not the linear
elevator. Position bounds and freshness/controller checks remain. Targets are never scaled,
clipped or silently altered. The former quarter-speed rejection threshold is removed.

The controller-manager loop is configured for 100 Hz; the companion timer is 20 Hz and
execute-mode joint feedback callbacks also monitor the limits. Completion is the scheduled
trajectory end plus final-hold publication, not proof of exact measured arrival. Preview
validates predictions without publishing trajectories or checking unexecuted tracking.

## Protocol

All three messages use `std_msgs/msg/String` over the existing Zenoh router (domain 30,
TCP 7447). No direct PRO 6000 connection is made by this companion.

- `/cosmos/bg2/chunk`: schema `cosmos_bg2_joint_chunk_v1`, execution contract
  `synchronous_full_chunk_v2`, current epoch/cycle, unique request ID, original observation
  timestamp, eight radian joint names/units, and exactly 16×8 absolute positions.
- `/cosmos/bg2/status`: schema `cosmos_bg2_status_v3`, execution contract, epoch, phase,
  integer cycle, `ready_since_ns`, `execution_end_ns`, remaining time, last submitted and
  completed request IDs, readiness, mode, controller health and diagnostic reason.
- `/cosmos/bg2/control`: schema `cosmos_bg2_control_v1`, current epoch, command `pause`,
  `resume`, or terminal `stop`. Pause/resume rotate epochs to reject late old commands.

Earlier deployment copies fail this contract. Update the gateway and companion together.
A missing chunk/completion acknowledgement leaves the gateway waiting, not requesting again.

## Run later on FFW

Reuse the installed `robotis/ai-worker:2.2.8` image in a separate container. No build,
package installation or image pull is needed. The existing ai_worker container is not
edited or restarted. Our entrypoint bypasses inherited s6/bringup and starts only Python.

After the approved copy, from this generated folder:

```bash
python3 verify_bundle.py
cp .env.example .env
# Configure the verified profile and exact ROBOTIS initial-position YAML in .env.
docker compose run --rm companion \
  --profile /config/profile.json \
  --initial-positions /config/initial_positions.yaml --check
docker compose up --no-build --pull never
```

Default is preview with current-pose initialization. Code/profile/home mounts and the root filesystem are read-only;
there are no serial devices, GPU, Docker socket or privileged mounts. The existing local
Zenoh router is reused. This container does not start ROBOTIS or homing.

The selected SNUPI file is `bg2_vr_startup_snupi.yaml`. Its head_joint1 target 0.78 exceeds
the installed Rev4 upper limit 0.6951, so current validation rejects it. The YAML and
limits remain unchanged. The operator-approved separate execution profile permits head_joint1 up to 0.78;
the ROBOTIS model and reference limits are not edited. See `docs/policy_execution.md`
in the main repository. Without that override, resolve the mismatch before startup. The reference profile
also remains unverified with execution disabled.

## Restart, rollback and validation boundary

Ordinary Ctrl+C on the execute gateway pauses at measured positions and leaves the
companion alive. Repeat the gateway command with a new prompt/output directory to resume
from fresh measurements; no copy, companion restart or homing is needed. Current-pose
initialization is the default even after a companion restart. Set
`COSMOS_INITIALIZATION=homing` to require fresh external homing explicitly.
Faults, terminal stop, or external rehoming invalidate the active session and cannot
auto-resume. Diagnose faults before restarting; current-pose mode does not demand homing.
Restarting PRO 6000 alone does not restart this process. Paused holds retain all motion,
feedback, ownership and controller checks.

To remove only this service, use `docker compose down --timeout 10` from its folder.
That leaves existing ROBOTIS services and the reused image intact. It does not undo
physical motion or stop separately managed robot bringup.

If feedback/ownership is unavailable, publishing a stop hold may be impossible; the last
finite trajectory can finish and the controller can retain its final target. This is not
torque disable or an independent emergency stop. Physical stop behavior, controller
interpolation/tracking, calibration and ROS integration still require commissioning.
