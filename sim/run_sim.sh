#!/bin/bash
# Launch the Go2 MuJoCo sim (DDS domain 0, interface lo) from the unitree_mujoco clone with our changes applied.
#   UNITREE_MUJOCO  path of the unitree_mujoco clone   (default ~/eng6010/unitree_mujoco)
#   VENV_SIM        venv with mujoco + unitree_sdk2py   (default ~/eng6010/.venv_sim)
# Default: GPU rendering (WSL2 d3d12 -> NVIDIA).  SIM_CPU=1 uses CPU rendering instead, which you want while a
# training run is using nearly all of a 4 GB GPU.
UNITREE_MUJOCO="${UNITREE_MUJOCO:-$HOME/eng6010/unitree_mujoco}"
VENV_SIM="${VENV_SIM:-$HOME/eng6010/.venv_sim}"
if [ -z "$SIM_CPU" ]; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
fi
cd "$UNITREE_MUJOCO/simulate_python" && exec "$VENV_SIM/bin/python" unitree_mujoco.py "$@"
