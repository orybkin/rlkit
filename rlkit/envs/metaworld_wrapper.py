import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image
import pathlib
from blox import rmap_list, rmap

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent / 'dreamerv2' ))

from environments import MultiTaskMetaWorld
from collections import OrderedDict
# from dreamerv2.environments import MultiTaskMetaWorld
# import dreamerv2.wrappers

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
    
    super().__init__(wrapped_env, 1)
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
    return self.render_goal()
  
  def set_env_state(self, _):
    pass

  def get_image(self, width, height):
    if self._goal_set:
      self._goal_set = False
      return self._skewfit_goal['image']
    
    # self._offscreen.render(width, width, -1)
    # image = np.flip(self._offscreen.read_pixels(width, width)[0], 1)
    return self._env.sim.render(self._width, self._width)
  
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
      return rmap(lambda x: np.asarray(x)[None], self.render_goal())
    
    import pdb; pdb.set_trace()
    goals = []
    for i in range(batch_size):
      goals.append(self.render_goal())
    goals = rmap_list(np.stack, goals)
    return goals