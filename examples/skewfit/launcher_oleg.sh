#!/bin/bash
#conda activate p2e
export PYOPENGL_PLATFORM=egl
export MUJOCO_RENDERER=egl
export MUJOCO_GL=egl
export GL_DEVICE_ID=$SLURM_STEP_GPUS
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_PRELOAD=
cd /mnt/beegfs/home/oleh/code/p2e/rlkit_repo/examples/skewfit/

export PYOPENGL_PLATFORM=glfw
export MUJOCO_RENDERER=glfw
export MUJOCO_GL=glfw
export DISPLAY=:0
Xvfb :0 -screen 0 1024x768x16 &
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

echo "Running $@"
eval python "$@" &

wait
echo "All finished"
