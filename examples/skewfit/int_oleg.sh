#!/usr/bin/env bash
srun --gpus 1 --qos kostas-high --partition kostas-compute --time 24:00:00 --pty bash
conda deactivate
conda activate p2e_rlkit_py35
export PYOPENGL_PLATFORM=egl
export MUJOCO_RENDERER=egl
export MUJOCO_GL=egl
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
cd /mnt/beegfs/home/oleh/code/p2e/rlkit_repo/examples/skewfit

#export PYOPENGL_PLATFORM=glfw
#export MUJOCO_RENDERER=glfw
#export MUJOCO_GL=glfw
#export DISPLAY=:0
#Xvfb :0 -screen 0 1024x768x16 &
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Launch command
cd /mnt/beegfs/home/oleh/code/p2e/rlkit_repo/examples/skewfit/
conda deactivate
conda activate p2e_rlkit_py35
srun --gpus 1 --qos kostas-med --partition kostas-compute --time 24:00:00