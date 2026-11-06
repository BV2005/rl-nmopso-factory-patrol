#!/usr/bin/env bash
# One-time set-up of the simulation stack the flights in results/flights/ were run with
# (Ubuntu 24.04): ROS 2 Jazzy, Gazebo Harmonic, PX4-Autopilot SITL at the tested commit, the
# Micro XRCE-DDS Agent, a colcon workspace with px4_msgs at the tested commit, pymavlink.
# Needs sudo and about 15 GB of disk; PX4's build takes 10-30 min.
#   bash gazebo/setup_sim.sh
# Environment overrides: PX4_DIR (default ~/PX4-Autopilot), PX4_ROS_WS (default ~/px4_ros_ws).
set -euo pipefail
PX4_COMMIT=cda3c3603f          # v1.18.0-beta1-334 (tested)
PX4_MSGS_COMMIT=ed9ddee        # PX4/px4_msgs (tested)
XRCE_TAG=v2.4.3
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WS="${PX4_ROS_WS:-$HOME/px4_ros_ws}"

# ROS 2 Jazzy (+ Gazebo Harmonic through ros-gz)
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  sudo apt-get update && sudo apt-get install -y software-properties-common curl git
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt-get update && sudo apt-get install -y ros-jazzy-ros-base ros-jazzy-ros-gz python3-colcon-common-extensions
fi

# PX4-Autopilot SITL (its ubuntu.sh also installs Gazebo Harmonic if missing)
if [ ! -d "$PX4_DIR" ]; then
  git clone --recursive https://github.com/PX4/PX4-Autopilot.git "$PX4_DIR"
fi
cd "$PX4_DIR"
git fetch --all --tags
git checkout "$PX4_COMMIT"
git submodule update --init --recursive
bash Tools/setup/ubuntu.sh --no-nuttx
make px4_sitl

# Micro XRCE-DDS Agent
if ! command -v MicroXRCEAgent > /dev/null; then
  tmp=$(mktemp -d)
  git clone -b "$XRCE_TAG" https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$tmp/agent"
  cmake -S "$tmp/agent" -B "$tmp/agent/build" && cmake --build "$tmp/agent/build" -j"$(nproc)"
  sudo cmake --install "$tmp/agent/build" && sudo ldconfig /usr/local/lib/
fi

# px4_msgs workspace
mkdir -p "$WS/src"
if [ ! -d "$WS/src/px4_msgs" ]; then
  git clone https://github.com/PX4/px4_msgs.git "$WS/src/px4_msgs"
fi
(cd "$WS/src/px4_msgs" && git fetch && git checkout "$PX4_MSGS_COMMIT")
set +u; source /opt/ros/jazzy/setup.bash; set -u
(cd "$WS" && colcon build --packages-select px4_msgs)

python3 -m pip install --user --break-system-packages pymavlink==2.4.49 numpy
echo "done: PX4_DIR=$PX4_DIR, PX4_MSGS_WS=$WS/install"
