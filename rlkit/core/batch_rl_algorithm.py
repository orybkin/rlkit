import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            train_goal_sampling_mode = self.expl_data_collector._goal_sampling_mode
            # Regardless of variant, the first exploration goals are collected with the prior
            # This is needed for HER training, which uses replay buffer goals. The first batch is collected when the buffer is empty. VAE prior is the only goal sampling mode that works in the code so we have to use it.
            self.expl_data_collector._goal_sampling_mode = 'vae_prior'
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            self.expl_data_collector._goal_sampling_mode = train_goal_sampling_mode

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            
            # env = self.eval_data_collector._env
            # env.goal_sampling_mode = self.eval_data_collector._goal_sampling_mode
            # o = env.reset()
            # if (o['image'] == o['image_goal']).mean() == 1.0:
            #     import pdb; pdb.set_trace()
            
            eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            from rlkit.core import logger
            import numpy as np
            import pathlib
            if epoch % 50 == 0:
                npy_path = pathlib.Path(logger._snapshot_dir) / ('eval_episodes_' + str(epoch) + '.npy')
                np.save(npy_path, eval_paths)
                import imageio
                videos = []
                for i in range(len(eval_paths)):
                    episode = eval_paths[i]['observations']
                    execution = np.stack([s['image'] for s in episode])
                    goal = episode[0]['image_goal']
                    video = np.concatenate((goal[None].repeat(execution.shape[0], 0), execution), 1)
                    videos.append(video)

                gif_path = pathlib.Path(logger._snapshot_dir) / ('eval_traj_' + str(epoch) + '.gif')
                imageio.mimwrite(gif_path, np.concatenate(videos, 2))
            
            # import pdb; pdb.set_trace()
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
