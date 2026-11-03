#!/usr/bin/env bash
# Spawns an unexpected static box into the running factory world (Gazebo Harmonic), e.g. while the
# drones fly, to test the global-local follower's avoidance and replanning.
#   bash scripts/spawn_obstacle.sh X Y [SX SY H] [NAME]      world metres; default 0.6 x 0.6 x 2.5 m
#   bash scripts/spawn_obstacle.sh -8.5 4.6                  # a pallet stack in drone 1's patrol leg
#   bash scripts/spawn_obstacle.sh -10.75 -2 2.1 0.6 3 aisle # closes the aisle between rack rows 1 and 2
# Remove it again: gz service -s /world/factory/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean \
#                    --timeout 2000 --req 'name: "NAME", type: MODEL'
set -euo pipefail
X=$1; Y=$2; SX=${3:-0.6}; SY=${4:-0.6}; H=${5:-2.5}; NAME=${6:-obstacle_$(date +%s)}
Z=$(python3 -c "print($H / 2)")
GEOM="<geometry><box><size>$SX $SY $H</size></box></geometry>"
SDF="<sdf version='1.9'><model name='$NAME'><static>true</static><pose>$X $Y $Z 0 0 0</pose><link name='link'>"
SDF+="<collision name='collision'>$GEOM</collision><visual name='visual'>$GEOM"
SDF+="<material><ambient>0.8 0.4 0.1 1</ambient><diffuse>0.8 0.4 0.1 1</diffuse></material></visual></link></model></sdf>"
gz service -s /world/factory/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 \
  --req "sdf: \"$SDF\", name: \"$NAME\", allow_renaming: true"
echo "spawned $NAME: $SX x $SY x $H m at ($X, $Y)"
