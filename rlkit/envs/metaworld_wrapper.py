import atexit
import functools
import sys
import threading
import traceback

import gym
import numpy as np
from PIL import Image

class MetaWorld:

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
    from mujoco_py import MjRenderContext
    import metaworld.envs.mujoco.sawyer_xyz as sawyer
      if wrapped_env == 'SawyerReachEnv':
        self._env = sawyer.SawyerReachPushPickPlaceEnv(task_type='reach')
      else:
        self._env = getattr(sawyer, wrapped_env)()
    self.transpose = transpose
    self.grayscale = grayscale
    self.normalize = normalize
    self.recompute_reward = recompute_reward
    self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
    self._width = imsize
    self._size = (self._width, self._width)

    self._offscreen = MjRenderContext(self._env.sim, True, 0, 'glfw', True)
    self._offscreen.cam.azimuth = 205
    self._offscreen.cam.elevation = -165
    self._offscreen.cam.distance = 2.6
    self._offscreen.cam.lookat[0] = 1.1
    self._offscreen.cam.lookat[1] = 1.1
    self._offscreen.cam.lookat[2] = -0.1

  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    self._env.reset()
    return self._get_obs()

  def step(self, action):
    _, reward, done, info = self._env.step(action)
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def render_goal(self):
    self._env.hand_init_pos = np.array([-0.1, 0.8, 0.2])
    self.reset()
    action = np.zeros(self._env.action_space.low.shape)
    self._env.step(action)
    goal_obs = self._get_obs()
    goal_obs['reward'] = 0.0
    self._env.hand_init_pos = self._env.init_config['hand_init_pos']
    self.reset()
    return goal_obs

  def _get_obs(self):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)
    return {'image': image}


class Collect:

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
      info['episode'] = episode
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
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs
