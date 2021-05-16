import atexit
import functools
import sys
import threading
import traceback
import io

import gym
import numpy as np
from PIL import Image
import pathlib
from blox import rmap_list, rmap
import datetime
import uuid
import random

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / 'dreamerv2' ))

from environments import MultiTaskMetaWorld
from envs.mt_dmc import MultiTaskDeepMindControl
from envs.d4rl_envs import KitchenEnv
# import rlkit.envs
# from rlkit.envs.dreamer_environments import MultiTaskMetaWorld
from collections import OrderedDict
# from dreamerv2.environments import MultiTaskMetaWorld
# import dreamerv2.wrappers



class SFMultiTaskKitchen(KitchenEnv):
  def __init__(
      self,
      wrapped_env,
      imsize=84,
      init_camera=None,
      transpose=False,
      grayscale=False,
      normalize=False,
      reward_type='wrapped_env',
      threshold=10,
      image_length=None,
      presampled_goals=None,
      non_presampled_goal_img_is_garbage=False,
      recompute_reward=True
  ):
    self.transpose = transpose
    self.grayscale = grayscale
    self.normalize = normalize
    self.recompute_reward = recompute_reward
    self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    
    super().__init__(wrapped_env, 2, (imsize, imsize), use_goal_idx=True)
    self._goal_set = False
  
  def set_to_goal(self, goal):
    self._goal_set = True
    self._skewfit_goal = goal
  
  def get_env_state(self):
    return None
  
  def get_diagnostics(self, *args, **kwargs):
    return OrderedDict()
  
  def get_goal(self):
    return {'image': self.render_goal()}
  
  def reset(self):
    self.set_goal_idx(random.randint(1, len(self.get_goals())) - 1)
    return super().reset()
  
  def set_env_state(self, _):
    pass
  
  def get_image(self, width, height):
    if self._goal_set:
      self._goal_set = False
      return self._skewfit_goal['image']
    
    return self.render_offscreen()
    # return self._env.sim.render(self._width, self._width)
  
  def compute_rewards(self, _, __):
    return np.zeros((1))
  
  def _get_obs(self, state=None):
    if state is None:
      assert self._goal_set
      self._goal_set = False
      return self._skewfit_goal

    obs = super()._get_obs(state)
    for goal_idx in range(len(self._env.goals)):
      obs['metric_success_task_relevant/goal_'+str(goal_idx)] = np.nan
      obs['metric_success_all_objects/goal_'+str(goal_idx)]   = np.nan

    task_rel_success, all_obj_success = self.compute_success(self._env.goal_idx)
    obs['metric_success_task_relevant/goal_' + str(self._env.goal_idx)] = task_rel_success
    obs['metric_success_all_objects/goal_' + str(self._env.goal_idx)] = all_obj_success
    return obs
  
  def sample_goals(self, batch_size):
    if batch_size == 1:
      return rmap(lambda x: np.asarray(x)[None], {'image': self.render_goal()})
    
    import pdb;pdb.set_trace()
    goals = []
    for i in range(batch_size):
      goals.append(self.render_goal())
    goals = rmap_list(np.stack, goals)
    return goals


class SFMultiTaskDeepMindControl(MultiTaskDeepMindControl):
  def __init__(
      self,
      wrapped_env,
      imsize=84,
      init_camera=None,
      transpose=False,
      grayscale=False,
      normalize=False,
      reward_type='wrapped_env',
      threshold=10,
      image_length=None,
      presampled_goals=None,
      non_presampled_goal_img_is_garbage=False,
      recompute_reward=True
  ):
    self.transpose = transpose
    self.grayscale = grayscale
    self.normalize = normalize
    self.recompute_reward = recompute_reward
    self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    
    super().__init__(wrapped_env, 2, (imsize, imsize), use_goal_idx=True)
    self._goal_set = False

  def set_to_goal(self, goal):
    self._goal_set = True
    self._skewfit_goal = goal

  def get_env_state(self):
    return None

  def get_diagnostics(self, *args, **kwargs):
    return OrderedDict()

  def get_goal(self):
    return {'image': self.render_goal()}

  def reset(self):
    self.set_goal_idx(random.randint(1, len(self.get_goals())) - 1)
    return super().reset()

  def set_env_state(self, _):
    pass

  def get_image(self, width, height):
    if self._goal_set:
      self._goal_set = False
      return self._skewfit_goal['image']
  
    return self.render_offscreen()
    # return self._env.sim.render(self._width, self._width)

  def compute_rewards(self, _, __):
    return np.zeros((1))

  def _get_obs(self, state=None):
    if state is None:
      assert self._goal_set
      self._goal_set = False
      return self._skewfit_goal
  
    return super()._get_obs(state)

  def sample_goals(self, batch_size):
    if batch_size == 1:
      return rmap(lambda x: np.asarray(x)[None], {'image': self.render_goal()})
  
    import pdb; pdb.set_trace()
    goals = []
    for i in range(batch_size):
      goals.append(self.render_goal())
    goals = rmap_list(np.stack, goals)
    return goals

  def _update_obs(self, obs):
    obs = super()._update_obs(obs)
  
    for i, goal in enumerate(self.goals):
      d = self.compute_reward(i)[1]
      for k, v in d.items():
        d[k] = np.nan
      obs.update(d)
      
    obs.update(self.compute_reward()[1])
  
    return obs
  

class SFMultiTaskMetaWorld(MultiTaskMetaWorld):

  def __init__(
      self,
      wrapped_env,
      imsize=84,
      init_camera=None,
      transpose=False,
      grayscale=False,
      normalize=False,
      reward_type='wrapped_env',
      threshold=10,
      image_length=None,
      presampled_goals=None,
      non_presampled_goal_img_is_garbage=False,
      recompute_reward=True
  ):
    self.transpose = transpose
    self.grayscale = grayscale
    self.normalize = normalize
    self.recompute_reward = recompute_reward
    self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    
    super().__init__(wrapped_env, 1, use_goal_idx=True)
    self._width = imsize
    self._size = (self._width, self._width)
    self._goal_set = False

  def set_to_goal(self, goal):
    self._goal_set = True
    self._skewfit_goal = goal
    
  def get_env_state(self):
    return None
  
  def get_diagnostics(self, *args, **kwargs):
    return OrderedDict()
  
  def get_goal(self):
    return {'image': self.render_goal()}
  
  def reset(self):
    self.set_goal_idx(random.randint(1, len(self.get_goals())) - 1)
    return super().reset()
  
  def set_env_state(self, _):
    pass

  def get_image(self, width, height):
    if self._goal_set:
      self._goal_set = False
      return self._skewfit_goal['image']
    
    return self.render_offscreen()
    # return self._env.sim.render(self._width, self._width)
  
  def compute_rewards(self, _, __):
    return np.zeros((1))
   
  def _get_obs(self, state=None):
    if state is None:
      assert self._goal_set
      self._goal_set = False
      return self._skewfit_goal
  
    obs = super()._get_obs(state)
    for goal_idx in range(len(self._env.goals)):
      obs['metric_reward/goal_' + str(goal_idx)] = np.nan
      obs['metric_success/goal_' + str(goal_idx)] = np.nan
      obs['metric_hand_distance/goal_' + str(goal_idx)] = np.nan
      obs['metric_obj1_distance/goal_' + str(goal_idx)] = np.nan
      obs['metric_obj2_distance/goal_' + str(goal_idx)] = np.nan
      
    obs = self._env.add_pertask_success(obs, self._env.goal_idx)
    return obs
    
  def sample_goals(self, batch_size):
    if batch_size == 1:
      return rmap(lambda x: np.asarray(x)[None], {'image': self.render_goal()})
    
    import pdb; pdb.set_trace()
    goals = []
    for i in range(batch_size):
      goals.append(self.render_goal())
    goals = rmap_list(np.stack, goals)
    return goals
  
  
def save_episodes(directory, episodes):
  directory = pathlib.Path(directory).expanduser()
  directory.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
  filenames = []
  for episode in episodes:
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
      np.savez_compressed(f1, **episode)
      f1.seek(0)
      with filename.open('wb') as f2:
        f2.write(f1.read())
    filenames.append(filename)
  return filenames



class CollectDataset:

  def __init__(self, env, callbacks=None, precision=32):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))

    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      episode['idx_repeated'] = np.ones(len(episode['reward']), dtype=np.int32)*self.get_goal_idx()
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()
