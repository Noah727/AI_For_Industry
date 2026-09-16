#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/kilted/setup.bash
ros2 bag info "$@"
