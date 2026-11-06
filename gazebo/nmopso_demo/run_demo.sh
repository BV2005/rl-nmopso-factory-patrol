#!/usr/bin/env bash
# NMOPSO factory patrol demo (branch nmopso-demo-factory2): the 30 x 20 m factory world
# (worlds/factory.sdf) + 3 stock PX4 x500 (simulated GPS) spawned side by side on a shared launch
# area + one patrol_follower.py per drone: transit to its zone, patrol the NMOPSO-planned loop,
# transit back, land (paths/patrol_paths.json). patrol_recorder.py records every drone at 10 Hz
# (drone_states.csv, events.csv); after the flight check_patrol.py and metrics_summary.py
# (metrics_summary.csv) run on the logs in run_logs/<run>/.
#
#   nmopso_demo/run_demo.sh                 headless, 1 lap per drone
#   HEADLESS=0 nmopso_demo/run_demo.sh      with the Gazebo GUI
#   LAPS=0 nmopso_demo/run_demo.sh          patrol until Ctrl-C
#   SPEED=1.5 nmopso_demo/run_demo.sh       patrol speed (m/s)
#   TOPVIEW=1 nmopso_demo/run_demo.sh       Gazebo GUI looking straight down on the hall
#   MODEL=x500_patrol nmopso_demo/run_demo.sh
#                                           the sensor drone (gazebo/models/x500_patrol: 2D lidar +
#                                           depth/RGB camera; install it with gazebo/install_patrol_drone.sh).
#                                           Headless runs render the sensors on the GPU (EGL); one
#                                           ros_gz_bridge per drone publishes /droneN/scan,
#                                           /droneN/depth/points (and /droneN/camera/image with RGB=1)
#   MODEL=x500_patrol SENSOR_CKPT=results/ckpt/vS_seed1.pt nmopso_demo/run_demo.sh
#                                           PATROL legs planned online from the sensors by the
#                                           sensor-input policy (sensor_follower.py) instead of flown
#                                           from the plan; transits as planned. BUDGET=0.6 threat budget,
#                                           CLOUD_FRAME=optical if the depth points are z-forward
#   MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt nmopso_demo/run_demo.sh
#                                           PATROL legs flown one vT waypoint at a time from the sensors
#                                           (vt_follower.py, rlpatrol/vt_pilot.py); transits as planned.
#                                           CLOUD_FRAME as above; summary: scripts/vt_flight_report.py <run dir>
#
# worlds/factory.sdf is the factory2 world (= worlds/factory2.sdf of the repository) as given (it already has the magnetometer, IMU,
# NavSat and air-pressure systems PX4 needs). PX4 is started like
# sim/launch/drone.sh but with PX4's stock gz_x500 airframe (4001) instead of the GPS-denied
# patrol airframe. px4_msgs comes from the main patrol_drone workspace (read only).
set -euo pipefail

DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DEMO/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
PX4_MSGS_WS="${PX4_MSGS_WS:-$HOME/px4_ros_ws/install}"   # a colcon workspace with px4_msgs (gazebo/setup_sim.sh)
PATHS="${PATHS:-$DEMO/paths/patrol_paths.json}"
LOGS="$DEMO/run_logs/$(date +%Y%m%d_%H%M%S)"
LAPS="${LAPS:-1}"
SPEED="${SPEED:-1.0}"
MODEL="${MODEL:-x500}"                       # x500 (stock, GPS) | x500_patrol (sensor drone)
SENSOR_CKPT="${SENSOR_CKPT:-}"               # a sensor_obs checkpoint: fly PATROL legs with sensor_follower.py
VT_CKPT="${VT_CKPT:-}"                       # a vT checkpoint: fly PATROL legs with vt_follower.py
if [ -n "$SENSOR_CKPT" ]; then
    SENSOR_CKPT="$(realpath "$SENSOR_CKPT")"
    [ "$MODEL" = "x500_patrol" ] || { echo "[run_demo] SENSOR_CKPT needs MODEL=x500_patrol"; exit 1; }
fi
if [ -n "$VT_CKPT" ]; then
    VT_CKPT="$(realpath "$VT_CKPT")"
    [ "$MODEL" = "x500_patrol" ] || { echo "[run_demo] VT_CKPT needs MODEL=x500_patrol"; exit 1; }
    [ -z "$SENSOR_CKPT" ] || { echo "[run_demo] set SENSOR_CKPT or VT_CKPT, not both"; exit 1; }
fi
mkdir -p "$LOGS"
cp "$PATHS" "$LOGS/patrol_paths.json"

set +u
source /opt/ros/jazzy/setup.bash
source "$PX4_MSGS_WS/setup.bash"
set -u

PIDS=()
cleanup() {
    echo "[run_demo] stopping..."
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
    pkill -f "px4_sitl_default/bin/px4" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f MicroXRCEAgent 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# homes from the plan (world ENU), one per drone
mapfile -t HOMES < <(python3 -c "
import json; d = json.load(open('$PATHS'))
for dr in sorted(d['drones'], key=lambda x: x['id']): print(dr['id'], dr['home'][0], dr['home'][1])")

echo "[run_demo] logs: $LOGS"
export GZ_SIM_RESOURCE_PATH="$DEMO/worlds:$REPO/models:$REPO/sim/models:$REPO/sim/worlds:$PX4_DIR/Tools/simulation/gz/models:$PX4_DIR/Tools/simulation/gz/worlds"
GZ_ARGS=(-r -v 1)
if [ "${TOPVIEW:-0}" = "1" ]; then GZ_ARGS+=(--gui-config "$DEMO/topview.gui.config")
elif [ "${HEADLESS:-1}" = "1" ]; then
    GZ_ARGS+=(-s)
    # rendered sensors (lidar, cameras) without a display: offscreen EGL rendering on the GPU
    if [ "$MODEL" != "x500" ]; then GZ_ARGS+=(--headless-rendering); fi
fi
if [ "$MODEL" = "x500_patrol" ] && [ ! -e "$PX4_DIR/Tools/simulation/gz/models/x500_patrol" ]; then
    echo "[run_demo] x500_patrol is not installed in PX4: bash $REPO/install_patrol_drone.sh"; exit 1
fi
gz sim "${GZ_ARGS[@]}" "$DEMO/worlds/factory.sdf" > "$LOGS/world.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 60); do gz topic -l 2>/dev/null | grep -q "/world/factory/clock" && break; sleep 1; done
echo "[run_demo] world up"

MicroXRCEAgent udp4 -p 8888 > "$LOGS/xrce_agent.log" 2>&1 &
PIDS+=($!)


for h in "${HOMES[@]}"; do
    read -r ID X Y <<< "$h"
    (
        cd "$PX4_DIR"
        PX4_GZ_WORLDS="$DEMO/worlds" PX4_GZ_WORLD=factory PX4_GZ_STANDALONE=1 \
        PX4_SIM_MODEL="gz_$MODEL" PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE="$X,$Y,0.2,0,0,0" \
        PX4_UXRCE_DDS_NS="drone$ID" \
        exec ./build/px4_sitl_default/bin/px4 -d -i "$ID"
    ) > "$LOGS/px4_drone$ID.log" 2>&1 &
    PIDS+=($!)
    echo "[run_demo] PX4 drone $ID at ($X, $Y)"
    sleep 4
done

# sensor bridges (x500_patrol): the Gazebo sensor topics of each drone -> ROS 2 /droneN/...
if [ "$MODEL" = "x500_patrol" ]; then
    for h in "${HOMES[@]}"; do
        read -r ID X Y <<< "$h"
        base="/world/factory/model/x500_patrol_$ID/link"
        scan="$base/link/sensor/lidar_2d_v2/scan"
        points="$base/camera_link/sensor/depth/depth_image/points"
        image="$base/camera_link/sensor/rgb/image"
        for _ in $(seq 60); do gz topic -l 2>/dev/null | grep -qx "$scan" && break; sleep 1; done
        if ! gz topic -l 2>/dev/null | grep -qx "$scan"; then
            echo "[run_demo] drone $ID: no $scan; its topics:"; gz topic -l | grep "x500_patrol_$ID" | head -20; exit 1
        fi
        gz topic -l 2>/dev/null | grep -qx "$points" || echo "[run_demo] drone $ID: WARNING no $points (lidar only)"
        cfg="$LOGS/bridge_drone$ID.yaml"
        {
            echo "- {ros_topic_name: /drone$ID/scan, gz_topic_name: $scan, ros_type_name: sensor_msgs/msg/LaserScan, gz_type_name: gz.msgs.LaserScan, direction: GZ_TO_ROS}"
            echo "- {ros_topic_name: /drone$ID/depth/points, gz_topic_name: $points, ros_type_name: sensor_msgs/msg/PointCloud2, gz_type_name: gz.msgs.PointCloudPacked, direction: GZ_TO_ROS}"
            if [ "${RGB:-0}" = "1" ]; then
                echo "- {ros_topic_name: /drone$ID/camera/image, gz_topic_name: $image, ros_type_name: sensor_msgs/msg/Image, gz_type_name: gz.msgs.Image, direction: GZ_TO_ROS}"
            fi
        } > "$cfg"
        ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:="$cfg" > "$LOGS/bridge_drone$ID.log" 2>&1 &
        PIDS+=($!)
        echo "[run_demo] sensor bridge drone $ID: /drone$ID/scan, /drone$ID/depth/points"
    done
fi

# 10 Hz state and event recorder (drone_states.csv, events.csv), started before the followers
python3 "$DEMO/patrol_recorder.py" --ros-args -p paths_file:="$PATHS" -p out_dir:="$LOGS" \
    > "$LOGS/recorder.log" 2>&1 &
RECORDER=$!
PIDS+=($RECORDER)

FOLLOWERS=()
for h in "${HOMES[@]}"; do
    read -r ID X Y <<< "$h"
    if [ -n "$VT_CKPT" ]; then
        PYTHONPATH="$REPO/..:$DEMO${PYTHONPATH:+:$PYTHONPATH}" python3 "$DEMO/vt_follower.py" --ros-args \
            -p drone_id:="$ID" -p paths_file:="$PATHS" -p log_file:="$LOGS/trace_drone$ID.csv" -p laps:="$LAPS" \
            -p speed:="$SPEED" -p ckpt:="$VT_CKPT" -p cloud_frame:="${CLOUD_FRAME:-flu}" \
            > "$LOGS/follower_drone$ID.log" 2>&1 &
    elif [ -n "$SENSOR_CKPT" ]; then
        PYTHONPATH="$REPO/..:$DEMO${PYTHONPATH:+:$PYTHONPATH}" python3 "$DEMO/sensor_follower.py" --ros-args \
            -p drone_id:="$ID" -p paths_file:="$PATHS" -p log_file:="$LOGS/trace_drone$ID.csv" -p laps:="$LAPS" \
            -p speed:="$SPEED" -p ckpt:="$SENSOR_CKPT" -p budget:="${BUDGET:-0.6}" \
            -p cloud_frame:="${CLOUD_FRAME:-flu}" > "$LOGS/follower_drone$ID.log" 2>&1 &
    else
        python3 "$DEMO/patrol_follower.py" --ros-args -p drone_id:="$ID" -p paths_file:="$PATHS" \
            -p log_file:="$LOGS/trace_drone$ID.csv" -p laps:="$LAPS" -p speed:="$SPEED" \
            > "$LOGS/follower_drone$ID.log" 2>&1 &
    fi
    FOLLOWERS+=($!)
    PIDS+=($!)
done
echo "[run_demo] 3 followers started; waiting for the patrol to finish (Ctrl-C to stop)"
for p in "${FOLLOWERS[@]}"; do wait "$p" || true; done
echo "[run_demo] patrol finished"
# the recorder stops by itself 3 s after every drone is DONE; SIGINT if it has not
for _ in $(seq 10); do kill -0 "$RECORDER" 2>/dev/null || break; sleep 1; done
kill -INT "$RECORDER" 2>/dev/null || true; wait "$RECORDER" 2>/dev/null || true
python3 "$DEMO/check_patrol.py" "$LOGS" | tee "$LOGS/summary.txt"
python3 "$DEMO/metrics_summary.py" "$LOGS" | tee -a "$LOGS/summary.txt"
