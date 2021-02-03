#!/bin/bash
#conda activate p2e
export PYOPENGL_PLATFORM=egl
export MUJOCO_RENDERER=egl
export MUJOCO_GL=egl
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
cd /mnt/beegfs/home/oleh/code/p2e/rlkit_repo/examples/skewfit/

echo "Running $@"
eval python "$@" &

wait
echo "All finished"
