#!/usr/bin/env bash
# Flies a patrol plan in Gazebo + PX4 SITL (3 x500, headless, one lap at 1 m/s) with the demo in
# gazebo/nmopso_demo, then runs the demo's checks (summary.txt) and the repository's independent
# collision check (flight_check.json) on the recorded 10 Hz states.
#   bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
#   HEADLESS=0 bash gazebo/fly_plan.sh <plan>        # with the Gazebo GUI (TOPVIEW=1: top view)
# Needs the stack of gazebo/setup_sim.sh (PX4_DIR, PX4_MSGS_WS as there); the collision check needs
# only numpy.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PLAN="$(realpath "$1")"
DEMO="$HERE/nmopso_demo"
mkdir -p "$DEMO/run_logs"
before=$(ls "$DEMO/run_logs" | sort | tail -1)
PATHS="$PLAN" HEADLESS="${HEADLESS:-1}" LAPS="${LAPS:-1}" SPEED="${SPEED:-1.0}" timeout 900 "$DEMO/run_demo.sh"
echo "run_demo exit $?"
run=$(ls "$DEMO/run_logs" | sort | tail -1)
[ "$run" = "$before" ] && { echo "no new run folder"; exit 1; }
echo "run folder: $DEMO/run_logs/$run"
python3 "$REPO/scripts/flight_check.py" "$DEMO/run_logs/$run" --out "$DEMO/run_logs/$run/flight_check.json" \
  | grep -E '"any_(obstacle|drone)_collision"'
