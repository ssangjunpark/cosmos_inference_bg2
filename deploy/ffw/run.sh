#!/bin/bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
exec python3 -B -m cosmos_ffw.node "$@"
