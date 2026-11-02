# Running the sensor-drone branch on a cloud GPU machine

Target: Ubuntu 24.04 with an NVIDIA GPU (driver installed, `nvidia-smi` works). The sensors are
rendered by Gazebo (ogre2) **headless on the GPU through EGL**. This is why the branch is tested on a
GPU machine: CPU-only rendering (for example WSL without a GPU) stalls the physics clock.

## 0. Get the branch

```bash
git clone https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
git checkout physical-ai-robust-rl
```

## 1. Python and the fast checks (minutes)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>   # the CUDA build for your driver (pytorch.org)
pip install numpy scipy matplotlib pyyaml
python scripts/test_sensors.py --device cuda     # the new code: sensor geometry, converters, env, training, planner
python scripts/smoke_test.py --device cuda       # the existing pipeline still passes (default observation unchanged)
```

`test_sensors.py` also prints how long `sim_scan` takes for 8192 drones: that is the per-step
overhead of sensor-input training.

## 2. Train the sensor-input policy

```bash
python scripts/train.py --name vS_seed1 --config configs/vS.yaml --seed 1
# for the paper: seeds 2 and 3 as well
python scripts/evaluate.py --ckpt results/ckpt/vS_seed1.pt --legs real              # ideal sensors
python scripts/evaluate.py --ckpt results/ckpt/vS_seed1.pt --legs real --sensor_noise --tag vS_seed1_real_noise
python scripts/evaluate.py --ckpt results/ckpt/vS_seed1.pt --legs blocked
python scripts/evaluate.py --ckpt results/ckpt/vB_seed1.pt --legs real               # the oracle baseline
```

GRU actor for obstacle robustness (memory of the earlier scans + 30 % blocked training legs):

```bash
python scripts/train.py --name vSG_seed1 --config configs/vSG.yaml --seed 1
python scripts/evaluate.py --ckpt results/ckpt/vSG_seed1.pt --legs blocked     # vS seed 1: HV 0.458, 184/200
python scripts/evaluate.py --ckpt results/ckpt/vSG_seed1.pt --legs real
# ablations (which half helps?): vS + blocked legs only, vS + GRU only
python scripts/train.py --name vSb_seed1 --config configs/vS.yaml --seed 1 --blocked_frac 0.3
python scripts/train.py --name vSr_seed1 --config configs/vS.yaml --seed 1 --recurrent true
```

The question to answer is how much HV and feasibility is lost when the policy sees only the 27-bin
scan instead of oracle geometry (vB). The privileged critic should close part of that gap; to measure
how much, train once with `--priv_critic false` as the ablation.

## 3. Simulation stack and the sensor drone

```bash
bash gazebo/setup_sim.sh       # ROS 2 Jazzy, Gazebo Harmonic, PX4 at the tested commit, ... + install_patrol_drone.sh
```

Check that the GPU renders headless:

```bash
# for containers: run with --gpus all and NVIDIA_DRIVER_CAPABILITIES=all (graphics, not just compute)
ls /usr/share/glvnd/egl_vendor.d/        # 10_nvidia.json must be there
```

## 4. Fly the planned paths with the sensor drone (checks the sensors, not the policy)

```bash
MODEL=x500_patrol bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
```

While it flies, in another shell (`source /opt/ros/jazzy/setup.bash`):

```bash
gz topic -e -t /world/factory/stats -n 1 | grep real_time_factor     # should stay near 1 (and must be > 0.5)
ros2 topic hz /drone1/scan                                          # ~30 Hz
ros2 topic hz /drone1/depth/points                                  # ~15 Hz
ros2 topic echo /drone1/scan --once | head -40                      # hovering in the open: no ranges < 0.3 m
```

**Verify these assumptions first.** They are written from the Gazebo and PX4 sources but have not
been run yet:

1. **Topic names.** `run_demo.sh` expects
   `/world/factory/model/x500_patrol_N/link/camera_link/sensor/depth/depth_image/points`. If it warns
   "no ... points", run `gz topic -l | grep x500_patrol_1` and fix the name in `run_demo.sh`.
2. **Point-cloud frame.** The follower assumes the depth points are x forward, y left, z up
   (`CLOUD_FRAME=flu`). If returns of a wall ahead land in the side bins, use `CLOUD_FRAME=optical`.
3. **Lidar self-hits.** The scan plane is 0.315 m above base_link, above props and camera. If
   `/drone1/scan` shows returns under 0.3 m while hovering in the open, raise the mast
   (`model.sdf` lidar include z, **and** `LIDAR_Z` in `rlpatrol/sensors.py`).
4. **Real-time factor.** If 3 drones are too heavy, try 1 drone first, or lower the depth resolution
   or rate in `model.sdf` (keep `DEPTH_W/H` in `sensors.py` in step; only the aspect ratio matters).

## 5. Closed loop: the policy flies from its sensors

```bash
MODEL=x500_patrol SENSOR_CKPT=results/ckpt/vS_seed1.pt bash gazebo/fly_plan.sh \
    gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
```

Outputs, in `gazebo/nmopso_demo/run_logs/<run>/`:

| File | Contents |
|---|---|
| `summary.txt` | `check_patrol.py`: PATROL cross-track error against the track planned online |
| `flight_check.json` | collisions against the true obstacle boxes |
| `trace_droneN_sensor.csv` | every node decision: scan, action, node, `ok` / `blocked` / `straight_to_goal` / `leg_skipped` |
| `follower_droneN.log` | warnings when a leg had no clear proposal |

Useful variations:
- `BUDGET=0.3`: a tighter threat budget.
- `LAPS=2`: each lap is planned afresh.
- Spawning an unexpected box in the running world (`gz service -s /world/factory/create ...`) is the
  real version of `scripts/replan_obstacle.py`: the drone has to see it and plan around it.
