#!/bin/bash
# Add our changes to a clone of https://github.com/unitreerobotics/unitree_mujoco (we started from commit 1eb6642).
#   sim/install_into_unitree_mujoco.sh [path to the unitree_mujoco clone]     (default ~/eng6010/unitree_mujoco)
# - applies unitree_mujoco_changes.patch (config: DDS domain 0 + obstacle scene + lidar switch, real-time-factor
#   print, body_height in sportmodestate, big-message DDS fix hook)
# - copies sim_lidar.py (simulated lidar) and the scene files into the clone
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/eng6010/unitree_mujoco}"
[ -d "$DEST/simulate_python" ] || { echo "not a unitree_mujoco clone: $DEST"; exit 1; }
if git -C "$DEST" apply --check "$HERE/unitree_mujoco_changes.patch" 2>/dev/null; then
  git -C "$DEST" apply "$HERE/unitree_mujoco_changes.patch"
  echo "patch applied"
else
  echo "patch does not apply cleanly (already applied, or the clone differs); skipping it"
fi
cp "$HERE/sim_lidar.py" "$DEST/simulate_python/"
cp "$HERE"/scene_*.xml "$DEST/unitree_robots/go2/"
echo "copied sim_lidar.py and scene_*.xml into $DEST"
