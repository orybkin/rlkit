#!/usr/bin/env bash
srun --gpus 1 --qos kostas-high --partition kostas-compute --time 24:00:00 --pty bash
conda deactivate
conda activate p2e_rlkit_py35
export PYOPENGL_PLATFORM=egl
export MUJOCO_RENDERER=egl
export MUJOCO_GL=egl
export GL_DEVICE_ID=$SLURM_STEP_GPUS
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
export LD_PRELOAD=
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so # using only this crashes bc of gl version
#export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so # using only this leads to black squares
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
# CRITICAL:absl:Shadow framebuffer is not complete, error 0x8cd7
#CRITICAL:absl:Could not allocate display lists
#CRITICAL:absl:Could not allocate display lists
#CRITICAL:absl:Could not allocate display lists
#CRITICAL:absl:Could not allocate font lists
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

srun --gpus rtx2080ti:1 --qos kostas-high --partition kostas-compute --time 7-00:00:00 launcher_oleg.sh dmc_walker.py



# Machine setup

# create environment.yml
conda env create -f environment.yml
conda deactivate
conda deactivate
conda activate p2e_rlkit_py35
pip install git+https://github.com/vitchyr/multiworld.git@f711cdb
pip install git+git://github.com/aravindr93/mjrl
pip install gym==0.10.5 opencv-python scikit-video numba gpustat

vim ~/.bashrc
# add these lines:
conda activate p2e_rlkit_py35
cd goalexploration
gpu() {
  export CUDA_VISIBLE_DEVICES=$1
}
export PYOPENGL_PLATFORM=egl
export MUJOCO_RENDERER=egl
export MUJOCO_GL=egl

git config --global credential.helper 'cache --timeout 86400'
git clone https://github.com/orybkin/goalexploration.git
git pull origin dev_oleg5
git submodule init
git submodule update
# After conda installed
python setup.py develop
cd metaworld/
python setup.py develop
cd ../dreamerv2/envs/d4rl_repo/
python setup.py develop
cd ../rlkit_repo
python setup.py develop

cd ~/goalexploration/rlkit_repo/examples/skewfit/
gpu 0
python mw_pickblock.py
