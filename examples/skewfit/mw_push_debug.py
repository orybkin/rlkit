import rlkit.util.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.metaworld_wrapper import SFMultiTaskMetaWorld
import multiworld.core.image_env

if __name__ == "__main__":
    ptu.set_gpu_mode(True)
    variant = dict(
        algorithm='Skew-Fit',
        double_algo=False,
        online_vae_exploration=False,
        imsize=48,
        # init_camera=sawyer_init_camera_zoomed_in,
        # env_id='SawyerPushNIPSEasy-v0',
        env_class = SFMultiTaskMetaWorld,
        env_kwargs = dict(wrapped_env='sawyer_SawyerPushEnvV2_gmed_ieasy_closeup', imsize=48),
        skewfit_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=150,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=1000,
                num_eval_steps_per_epoch=500,
                num_expl_steps_per_train_loop=500,
                num_trains_per_train_loop=1000,
                min_num_steps_before_training=10000,
                vae_training_schedule=vae_schedules.custom_schedule_2,
                oracle_data=False,
                vae_save_period=50,
                parallel_vae_train=False,
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=-1,
                relabeling_goal_sampling_mode='vae_prior',
            ),
            exploration_goal_sampling_mode='vae_prior',
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False,
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=40,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                oracle_dataset_using_set_to_goal=True,
                n_random_steps=100,
                non_presampled_goal_img_is_garbage=True,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize48_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            # TODO: why the redundancy?
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=-1,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
    )
    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'rlkit-skew-fit-mwpush'

    from rlkit.samplers.data_collector.vae_env import VAEWrappedEnvPathCollector
    from rlkit.launchers.skewfit_experiments import get_envs, skewfit_preprocess_variant, full_experiment_variant_preprocess, train_vae_and_update_variant
    from rlkit.policies.simple import RandomPolicy
    from blox import AttrDict
    from rlkit.samplers.data_collector.path_collector import MdpPathCollector

    variant['skewfit_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    # train_vae_and_update_variant(variant)
    skewfit_preprocess_variant(variant['skewfit_variant'])
    variant['skewfit_variant']['vae_path'] = AttrDict(representation_size=10, input_channels=3)
    env = get_envs(variant['skewfit_variant'])
    eval_path_collector = VAEWrappedEnvPathCollector(
        variant['skewfit_variant']['evaluation_goal_sampling_mode'],
        env.wrapped_env,
        RandomPolicy(env.action_space),
        50,
        observation_key='image',
        desired_goal_key='image_goal',
    )

    eval_paths = eval_path_collector.collect_new_paths(50, 150, discard_incomplete_paths=True)
    # MdpPathCollector.collect_new_paths(eval_path_collector, 50, 3, discard_incomplete_paths=True)
    from rlkit.core import logger
    import numpy as np
    import pathlib
    np.save('eval_episodes_test.npy', eval_paths)
    
    import pdb; pdb.set_trace()

    # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
    #     for _ in range(n_seeds):
    #         run_experiment(
    #             skewfit_full_experiment,
    #             exp_prefix=exp_prefix,
    #             mode=mode,
    #             variant=variant,
    #             use_gpu=True,
    #             num_exps_per_instance=3,
    #             gcp_kwargs=dict(
    #                 terminate=True,
    #                 zone='us-east1-c',
    #                 gpu_kwargs=dict(
    #                     gpu_model='nvidia-rtx-2080ti',
    #                     num_gpu=1,
    #                 )
    #             )
    #       )
