#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/kilted/setup.bash
source /ws_aic/install/setup.bash
python3 "$@"
