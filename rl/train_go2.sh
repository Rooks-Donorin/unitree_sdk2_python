#!/bin/bash
# Train the Go2 walking policy (Isaac Gym + rsl_rl PPO, headless).  Usage: rl/train_go2.sh [iterations] [run_name]
#   UNITREE_RL_GYM  path of the unitree_rl_gym clone   (default ~/eng6010/unitree_rl_gym)
#   RL_CONDA_ENV    conda env with isaacgym, rsl_rl, torch (default ~/miniconda3/envs/unitree-rl)
# Checkpoints land in $UNITREE_RL_GYM/logs/rough_go2/<date>_<run_name>/model_<iter>.pt every 50 iterations.
# Task "go2" is Unitree's unmodified Go2 config on a flat plane (mesh_type = 'plane').
E="${RL_CONDA_ENV:-$HOME/miniconda3/envs/unitree-rl}"
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
cd "${UNITREE_RL_GYM:-$HOME/eng6010/unitree_rl_gym}" && exec python legged_gym/scripts/train.py --task=go2 --headless \
    --max_iterations="${1:-1500}" --run_name="${2:-walk}"
