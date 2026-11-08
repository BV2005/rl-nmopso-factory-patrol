#!/usr/bin/env bash
# Installs the x500_patrol sensor drone into the simulation stack of gazebo/setup_sim.sh:
#   * links gazebo/models/x500_patrol into PX4's model folder (PX4 spawns models from there only:
#     build/px4_sitl_default/rootfs/gz_env.sh fixes PX4_GZ_MODELS); PX4's stock airframe 4001 then
#     flies it with PX4_SIM_MODEL=gz_x500_patrol (run_demo.sh MODEL=x500_patrol), no PX4 rebuild;
#   * the ROS 2 side of the sensors: ros_gz_bridge and sensor_msgs_py;
#   * PyTorch etc. for the system python3 that runs sensor_follower.py (CPU wheels: the policy is a
#     small MLP, one forward pass per node);
#   * checks that the model parses.
#   bash gazebo/install_patrol_drone.sh            (PX4_DIR as in setup_sim.sh, default ~/PX4-Autopilot)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
MODELS="$PX4_DIR/Tools/simulation/gz/models"
[ -d "$MODELS" ] || { echo "no PX4 model folder $MODELS (run gazebo/setup_sim.sh first)"; exit 1; }

ln -sfn "$HERE/models/x500_patrol" "$MODELS/x500_patrol"
echo "linked $MODELS/x500_patrol -> $HERE/models/x500_patrol"

if ! dpkg -s ros-jazzy-ros-gz-bridge > /dev/null 2>&1 || ! dpkg -s ros-jazzy-sensor-msgs-py > /dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y ros-jazzy-ros-gz-bridge ros-jazzy-sensor-msgs-py
fi

if [ "${NO_PIP:-0}" != 1 ] && ! python3 -c "import torch, scipy" > /dev/null 2>&1; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then PIP=(python3 -m pip install); else PIP=(python3 -m pip install --user --break-system-packages); fi
  "${PIP[@]}" -r "$REPO/requirements.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

# check that the model parses: the standalone `gz sdf` resolves the included models (x500, lidar_2d_v2) through
# SDF_PATH, not GZ_SIM_RESOURCE_PATH as Gazebo does; a failure here is a warning (Gazebo's own spawn in run_demo.sh,
# which checks the sensor topics, is the real test)
export GZ_SIM_RESOURCE_PATH="$MODELS${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
if SDF_PATH="$MODELS${SDF_PATH:+:$SDF_PATH}" gz sdf -k "$HERE/models/x500_patrol/model.sdf" > /tmp/x500_patrol_sdf_check.log 2>&1; then
  echo "x500_patrol model.sdf: valid"
else
  echo "WARNING: 'gz sdf -k' could not fully check x500_patrol (see /tmp/x500_patrol_sdf_check.log); continue with"
  echo "         MODEL=x500_patrol bash gazebo/fly_plan.sh <plan> - it fails early if Gazebo cannot spawn the drone"
fi
echo "done: fly it with  MODEL=x500_patrol bash gazebo/fly_plan.sh <plan>"
