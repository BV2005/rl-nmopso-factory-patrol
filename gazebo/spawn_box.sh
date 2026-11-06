#!/usr/bin/env bash
# Drops an unexpected obstacle (a static box on the floor) into the running factory world: the drones must see it and
# plan around it (in no map: vt_follower.py / sensor_follower.py only sense it). Records it in the run folder's
# extra_boxes.json, so scripts/flight_check.py counts collisions with it too.
#   bash gazebo/spawn_box.sh <run dir> <x> <y> [size_x 0.6] [size_y 0.6] [height 2.5] [name]
# e.g. while a flight runs (the newest run folder):
#   bash gazebo/spawn_box.sh "$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)" -10.0 3.65
set -euo pipefail
RUN="$1"; X="$2"; Y="$3"; SX="${4:-0.6}"; SY="${5:-0.6}"; H="${6:-2.5}"
NAME="${7:-unexpected_box_$(date +%H%M%S)}"
ZC=$(python3 -c "print($H / 2)")
SDF="<?xml version='1.0'?><sdf version='1.9'><model name='$NAME'><static>true</static><pose>$X $Y $ZC 0 0 0</pose><link name='link'>"
SDF+="<collision name='c'><geometry><box><size>$SX $SY $H</size></box></geometry></collision>"
SDF+="<visual name='v'><geometry><box><size>$SX $SY $H</size></box></geometry><material><diffuse>0.9 0.4 0.1 1</diffuse></material></visual>"
SDF+="</link></model></sdf>"
REQ="sdf: \"$(printf '%s' "$SDF" | sed 's/"/\\"/g')\""
gz service -s /world/factory/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 --req "$REQ"
python3 - "$RUN" "$X" "$Y" "$SX" "$SY" "$H" "$NAME" <<'PY'
import json, sys
from pathlib import Path
run, x, y, sx, sy, h, name = sys.argv[1], *map(float, sys.argv[2:7]), sys.argv[7]
p = Path(run) / "extra_boxes.json"
boxes = json.loads(p.read_text()) if p.exists() else []
boxes.append({"name": name, "box": [x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2, h]})
p.write_text(json.dumps(boxes, indent=1))
print(f"spawned {name} at ({x}, {y}), {sx} x {sy} x {h} m; recorded in {p}")
PY
