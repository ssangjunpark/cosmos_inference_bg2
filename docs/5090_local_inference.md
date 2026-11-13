# Cosmos3 Edge: start, stop, and reset

Use four terminals on **5090**: model, follower, companion, and gateway.
From another computer, connect each terminal with `ssh 3sp5090`.
The commands below use the installed ROBOTIS environment on FFW, with its
`ai_worker` container and Zenoh router running.

Run commands on the host unless a step explicitly says **inside `ai_worker`**.
Conda `(base)` can remain active. The model launcher starts Docker automatically.

For a fresh restart of an existing session, first follow **Stop and reset** below.

## 1. Start the model — terminal 1, on 5090

```bash
bash /mnt/sata1/cosmos3/runtime/deploy/5090/run_edge.sh --allow-busy-gpu
```

Wait for `"event": "ready"`. Keep this terminal open.

## 2. Start the follower and cameras — terminal 2

From **5090**, connect to the **FFW host** and enter its existing container:

```bash
ssh ffw
docker exec -it ai_worker bash
```

Now **inside `ai_worker`**, start the follower:

```bash
ros2 launch ffw_bringup ffw_bg2_follower_ai.launch.py \
  launch_cameras:=true \
  init_position:=true \
  init_position_file:=bg2_vr_startup_snupi.yaml \
  use_head_eef_tracker:=false
```

This moves the robot to the SNUPI starting pose and starts the cameras and joint
feedback. Wait for homing to finish and the cameras to start before continuing.
Keep this terminal open. Run only one follower launch, and keep the Cosmos
companion stopped during homing.

## 3. Start the FFW companion — terminal 3

Open another terminal on **5090**:

```bash
ssh ffw
cd ~/cosmos_ffw_companion
FFW_PROFILE=./config/profile.execute.json \
COSMOS_MODE=execute COSMOS_INITIALIZATION=current \
  docker compose up --no-build --pull never
```

These commands run on the **FFW host**, outside `ai_worker`.
The companion captures and holds the current robot pose. Keep this terminal open.

## 4. Start inference and robot execution — terminal 4, on 5090

Start this after the model is ready and the follower and companion are running.

```bash
cd ~/cosmos_bg2_gateway
COSMOS_PROFILE=./config/profile.execute.json \
docker compose -f compose.yaml \
  -f /mnt/sata1/cosmos3/runtime/deploy/5090/gateway.local.yaml \
  run --rm --no-deps --name cosmos-bg2-execute gateway \
  execute --profile /config/profile.json \
  --messages-dir /opt/cosmos_gateway/messages \
  --server-url http://127.0.0.1:18008 --timeout 120 \
  --instruction "Pick up the police car and place it on the top-right picture" \
  --output "/output/execute-$(date -u +%Y%m%dT%H%M%SZ)"
```

This starts robot execution. Change `--instruction` to your task.
Recordings are saved under `/mnt/sata1/cosmos3/gateway-logs/` on 5090.

## Pause or change the task

1. Press **Ctrl+C** in the gateway terminal (terminal 4).
2. Wait for `pause_confirmed` and recording to finish.
3. Repeat step 4 with the new instruction.

Keep the model, follower, and companion running during an ordinary task change.

## Stop and reset

1. **Gateway, terminal 4:** press **Ctrl+C**, wait for `pause_confirmed`, and let
   recording finish.
2. **Companion:** open another terminal on 5090 and run:

   ```bash
   ssh ffw
   cd ~/cosmos_ffw_companion
   docker compose down --timeout 3
   ```

3. **Follower, terminal 2:** press **Ctrl+C** in the original follower launch
   terminal inside `ai_worker`. Leave the `ai_worker` container running.
4. **Model, terminal 1:** press **Ctrl+C**. Alternatively, on 5090:

   ```bash
   docker stop --timeout 20 cosmos-bg2-wam-edge-server
   ```

To start again from the SNUPI starting pose, repeat steps **1 → 2 → 3 → 4**.
