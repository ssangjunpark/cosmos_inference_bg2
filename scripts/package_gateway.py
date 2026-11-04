#!/usr/bin/env python3
"""Prepare a small 5090 folder locally. No SSH, transfer, model weights, or ROS installation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SDK_COMMIT="be2c4d4595305a9c282fca09820a8b3bfb8076a3"


def build(output):
    output=Path(output)
    if output.exists():raise ValueError("Use a new output directory; earlier bundles are preserved")
    sdk=ROOT/"third_party/zenoh_ros2_sdk"
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=sdk,text=True).strip()
    if commit!=SDK_COMMIT:raise ValueError("Gateway requires the pinned ROBOTIS SDK revision")
    if subprocess.check_output(["git","diff","HEAD","--"],cwd=sdk,text=True).strip():
        raise ValueError("Gateway SDK has local source changes")
    output.mkdir(parents=True)
    def copy(source,relative):
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    templates=("Dockerfile", ".dockerignore", ".env.example", "compose.yaml", "README.md", "verify_bundle.py", "requirements.lock",
               "messages/SOURCE.json", "messages/NOTICE.md", "messages/LICENSE.Apache-2.0",
               "messages/builtin_interfaces/package.xml", "messages/std_msgs/package.xml", "messages/sensor_msgs/package.xml",
               "messages/builtin_interfaces/msg/Time.msg", "messages/std_msgs/msg/Header.msg", "messages/std_msgs/msg/String.msg",
               "messages/sensor_msgs/msg/CompressedImage.msg", "messages/sensor_msgs/msg/JointState.msg")
    for name in templates:copy(ROOT/"deploy/gateway"/name,Path(name))
    for name in ("gateway.py","bg2_inference.py"):copy(ROOT/"scripts"/name,Path("scripts")/name)
    for name in ("__init__.py","core.py","chunks.py","observations.py","rpc.py","gateway.py"):
        copy(ROOT/"scripts/bg2_bridge"/name,Path("scripts/bg2_bridge")/name)
    copy(ROOT/"profiles/robotis_bg2_rev4.reference.json",Path("config/profile.reference.json"))
    for name in ("LICENSE","README.md","pyproject.toml","setup.py"):
        copy(sdk/name,Path("third_party/zenoh_ros2_sdk")/name)
    # Copy only pinned tracked package files; exclude local caches, credentials and build artifacts.
    tracked=subprocess.check_output(["git","ls-files","-z","--","zenoh_ros2_sdk"],cwd=sdk,text=True)
    for name in tracked.split("\0"):
        if name:copy(sdk/name,Path("third_party/zenoh_ros2_sdk")/name)
    files={}
    for path in sorted(output.rglob("*")):
        if path.is_file():files[path.relative_to(output).as_posix()]={"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={"schema":1,"purpose":"5090 observation/inference gateway; prepared locally, not commissioned",
              "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "source_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
              "default_mode":"preview","sdk_commit":SDK_COMMIT,"files":files}
    (output/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"transfer/5090")
    args=parser.parse_args();result=build(args.output)
    print(json.dumps({"folder":str(args.output),"files":len(result["files"]),"bytes":sum(v["bytes"] for v in result["files"].values()),"transferred":False}))
