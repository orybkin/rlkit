import numpy as np
# import torch
import imageio

# data = np.load('../../data/01-26-rlkit-skew-fit-mwpush/01-26-rlkit-skew-fit-mwpush_2021_01_26_17_37_43_0000--s-51961/eval_episodes_100.npy', allow_pickle=True)
# data = np.load('../../data/01-29-rlkit-skew-fit-mwreach/01-29-rlkit-skew-fit-mwreach_2021_01_29_23_21_58_0000--s-71642/eval_episodes_0.npy', allow_pickle=True)
data = np.load('../../data/01-29-rlkit-skew-fit-mwreach/01-29-rlkit-skew-fit-mwreach_2021_01_29_23_26_59_0000--s-79310/eval_episodes_0.npy', allow_pickle=True)
# data = np.load('../../data/01-29-rlkit-skew-fit-mwreach/01-29-rlkit-skew-fit-mwreach_2021_01_29_20_35_42_0000--s-3291/eval_episodes_0.npy', allow_pickle=True)
# data = np.load('eval_episodes_0.npy')
# data = np.load('eval_episodes_test.npy', allow_pickle=True)

videos = []
print(data[0]['actions'])
for i in range(len(data))[:6]:
  episode = data[i]['observations']
  episode[0]['image'] == episode[0]['image_goal'].mean()
  execution = np.stack([s['image'] for s in episode])
  execution_goal = np.stack([s['image_desired_goal'] for s in episode]).reshape((-1, 3, 48, 48)).transpose([0, 3, 2, 1]) * 255
  # episodep = data[i]['next_observations']
  # executionp = np.stack([s['image'] for s in episodep])
  # execution = np.stack([s['image_observation'] for s in episode]).reshape((-1, 3, 48, 48)).transpose([0, 3, 2, 1]) * 255
  # execution = np.stack([s['image_achieved_goal'] for s in episode]).reshape((-1, 3, 48, 48)).transpose([0, 3, 2, 1]) * 255
  goal = episode[0]['image_goal']
  video = np.concatenate((goal[None].repeat(execution.shape[0],0), execution, execution_goal), 1)
  videos.append(video)
  
  
imageio.mimwrite('trajectory.gif', np.concatenate(videos, 2))