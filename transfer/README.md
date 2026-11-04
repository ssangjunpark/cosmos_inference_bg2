# Folders to copy to the other machines

| Generated folder | Send to | What it contains |
| --- | --- | --- |
| `ffw/` | FFW robot computer | Companion Python code, shared validation modules, Compose launcher and reference profile. Reuses the installed ROBOTIS image; no image build. |
| `5090/` | 5090 communication PC | Gateway Python code, pinned SDK, message definitions, dependency lock, Docker/Compose files and reference profile. |

Copy the complete named folder, including hidden files, when deployment is authorized.
Nothing in this directory is automatically transferred or started. The server token is
not included: it must be supplied separately to the 5090, with restrictive permissions.
Live camera/controller configuration still needs verification; preview is the default.

## Where to edit versus what to copy

Edit source under `deploy/ffw/`, `deploy/gateway/` and `scripts/`. Those files are
tracked by Git. Packaging combines machine-specific files with their required shared
modules into the complete folders above. The generated copies are ignored by Git;
changes there do not update the source and will not appear in a clone.

After cloning with the pinned SDK submodule, generate the folders from the repository root:

```bash
python3 scripts/package_ffw.py
python3 scripts/package_gateway.py
```

The scripts default to `transfer/ffw/` and `transfer/5090/`. They refuse an existing
output folder to preserve edits and earlier versions. To rebuild, preserve/move the
old folder first or choose a fresh path with `--output`.

Each folder has `MANIFEST.json` and `verify_bundle.py` for checking its files offline.
`outputs/` is reserved for inference results and measurements. Previous generated
bundle versions are archived locally under `.runtime/bundle_archive/`.

Use the [three-machine runbook](../docs/three_machine_pipeline.md) for configuration,
startup order, ports, preview and remaining physical commissioning checks.
