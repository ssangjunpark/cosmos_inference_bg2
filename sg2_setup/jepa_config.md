# One config for Cosmos CM1, CM2, JEPA and threshold

Use `run_jepa.sh` to launch the system from a single **host-path** config. The
same config can start services, start a gateway, or start both. This launcher
normalizes paths, saves a per-run config snapshot under `.runtime/sg2-jepa/`,
and checks the checkpoint paths advertised by all three services before starting
a gateway. The [dual-model guide](jepa.md) describes inference and switch behavior.

## Configure once

From the Cosmos repository:

```sh
cp configs/sg2_jepa.example.json configs/sg2_jepa.local.json
```

Local `*.local.json` configs are gitignored. Edit these four settings for each
experiment (the fragment below shows the fields, not a complete config):

```json
{
  "models": [
    {"model_id": "CM1", "checkpoint": "~/checkpoints/my-cosmos-1", "port": 8090},
    {"model_id": "CM2", "checkpoint": "~/checkpoints/my-cosmos-2", "port": 8091}
  ],
  "jepa": {
    "checkpoint": "~/checkpoints/vjepa2_1_vitg_384.pt"
  },
  "threshold": 2.0
}
```

For the initial installation, also fill the following in the complete example:

- `primary_variant` and `fallback_variant`: variants matching the two publications.
- `model_runtime`: tokenizer, VAE, verified calibration and sampling settings.
  `steps`, `guidance`, `shift` must come from your evaluation setup; nulls reject launch.
- `jepa.python`: the Python executable in the environment with JEPA dependencies.
  `jepa.project_root`: sibling `project-lg-wm` checkout. `jepa.repo_root`: external
  upstream V-JEPA checkout. `size`, `model_family` and preprocessing must match
  the JEPA weights. Teacher-space models additionally need `teacher_checkpoint`.
- `launcher`: Cosmos defaults to the existing `cosmos-sg2-wam:local` Docker image
  and GPU 0. `cosmos_backend: native` uses `cosmos_python` instead.
- `gateway`: installation router, ROS domain, three camera topics and profile.
  Default mode is `preview`; `chunks: 0` keeps demonstrating until interrupted.

Paths use `~`, environment variables or paths relative to the **config file**,
regardless of the shell's working directory. The template's relative paths assume
it stays in this repository's `configs/` directory. All paths in this launcher
are **host paths**, including with Docker: it binds asset parents at the same
absolute paths inside the container. Do not change them to `/data/...` for this
launcher. Keep asset symlink targets within the mounted asset parents. The older
`run_policy.sh` entry point retains its separate `/data` convention.

JEPA runs in its own host Python environment; Cosmos runs in Docker by default.
The JEPA example points to `project-lg-wm/.venv/bin/python`; replace it with your
actual configured interpreter if that environment lives elsewhere. Both may use
the same GPU, subject to real-weight memory measurements.

## Start with one config path

Inspect commands first; dry-run loads no weights, starts no processes and creates
no run directory. Checkpoint files may still be transferring:

```sh
bash sg2_setup/run_jepa.sh configs/sg2_jepa.local.json services --dry-run
```

Terminal A starts both resident Cosmos models and JEPA, then waits until all
three services report ready:

```sh
bash sg2_setup/run_jepa.sh configs/sg2_jepa.local.json
# Equivalent: append "services"
```

Start the robot companion in preview using the existing commissioning procedure.
Then terminal B starts the gateway using the same config:

```sh
bash sg2_setup/run_jepa.sh configs/sg2_jepa.local.json gateway
```

Alternatively, after the companion is ready, one command starts the services,
waits for readiness, then starts the gateway:

```sh
bash sg2_setup/run_jepa.sh configs/sg2_jepa.local.json all
```

`gateway.mode: execute` explicitly selects physical execution with an execute-mode
companion and reviewed robot profile. The launcher does not start the robot's ROS
bringup or companion. `services` starts only inference services.

The launcher stops only its own processes on Ctrl+C. The `gateway` action leaves
separately started services running. The `all` action stops its services when its
gateway finishes. After startup, a JEPA process failure preserves the Cosmos
processes, allowing an already latched CM2 gateway to continue. The Cosmos pair
supervisor likewise preserves CM2 if CM1 exits.

## Change checkpoints or threshold

- **Threshold only:** edit `threshold`, stop and restart the **gateway** with the
  same config. Keep terminal A's models resident. Restarting resets CM2 mode to CM1.
- **Any checkpoint:** stop the gateway and services, edit the checkpoint field(s),
  restart services, then restart the gateway. Checkpoint path mismatches fail
  before a gateway starts. Files modified at an unchanged path also require restart.
- **Multiple configs:** copy `configs/sg2_jepa.local.json` to another `*.local.json`
  experiment config, then pass that path to the script. Concurrent service sets
  need distinct ports and appropriate GPU capacity.

Latest dashboard: the host directory set by `gateway.selection_output`, containing
`index.html`, `report.json` and `events.jsonl`. The effective configs for each run
are retained under `.runtime/sg2-jepa/<run-id>/gate.json` and `jepa.json`.

## Validation

Unified-launcher tests cover the three checkpoint paths and threshold, relative
and home paths, paths containing spaces, virtualenv interpreter symlinks,
Docker mounts, separate service/gateway actions, startup checkpoint identity
checks, dry-run from another working directory and service-failure isolation.
The focused inference suites passed 55 tests (2 optional skips); sibling JEPA
suites passed 55 tests (1 upstream-checkout skip). These used CPU fixtures and
local HTTP services; real checkpoint/GPU/robot execution remains unmeasured.
