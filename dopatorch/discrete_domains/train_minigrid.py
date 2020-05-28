# =============================================================================
# Training the agent in MiniGrid environment
# Taken from:
# https://github.com/im-ant/RL-767/blob/master/flat_SMiRL/train_minigrid.py
#
# Potential Resources:
# - Some baseline plots from Google Dopamine:
#   https://google.github.io/dopamine/baselines/plots.html
# - A discussion on (the lack of) frame maxing:
#   https://github.com/openai/gym/issues/275
# - The DQN hyper-parameters, as reported by Google Dopamine:
#   https://github.com/google/dopamine/tree/master/dopamine/agents/dqn/configs
# - Saving images:
#   save_image(torch.tensor(tmp_obs, dtype=float).unsqueeze(0),
#                f'tmp_img{args.env_name}.png', normalize=True, range=(0, 255))
#
# NOTE:
# - agent configuration is acquired in a .ini file using configparser
#   - The choice of configparser was made over gin or yaml since it is
#     already present in python
#
# Author: Anthony G. Chen
# =============================================================================

import argparse
import configparser
import math
import random           # only imported to set seed
import sys

import gym
from gym_minigrid import wrappers
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from dopatorch.agents.a2c.a2c_agent import A2CAgent
from dopatorch.discrete_domains.minigrid_wrapper import MiniGridFlatWrapper


# ========================================
# Helper methods for agent initialization
# ========================================

def init_agent(args: argparse.Namespace, env, device='cpu'):
    """
    Method for initializing the agent
    :param args: argparse arguments
    :return: agent instance
    """

    # ===
    # Configuration parsing
    config = configparser.ConfigParser()
    config.read(args.config_path)

    # ===
    # Initialize agent
    if args.agent_type == 'a2c':
        agent = A2CAgent(
            action_space=env.action_space,
            observation_shape=env.observation_space.shape,
            observation_dtype=torch.float,
            gamma=config['Agent'].getfloat('gamma'),
            use_recurrent_net=config['Agent'].getboolean('use_recurrent_net'),
            num_rollout_steps=config['Agent'].getint('num_rollout_steps'),
            value_loss_coef=config['Agent'].getfloat('value_loss_coef'),
            entropy_coef=config['Agent'].getfloat('entropy_coef'),
            max_grad_norm=config['Agent'].getfloat('max_grad_norm'),
            use_acktr=config['Agent'].getboolean('use_acktr'),
            device=device
        )
    else:
        raise NotImplementedError

    return agent


# ========================================
# Run environment
# ========================================

def run_environment(args: argparse.Namespace,
                    device: str = 'cpu',
                    logger: torch.utils.tensorboard.SummaryWriter = None):
    # =========
    # Set up environment
    env = gym.make(args.env_name)
    env = MiniGridFlatWrapper(env, use_tensor=False,
                              scale_observation=False,
                              scale_min=0, scale_max=10)

    # =========
    # Set up agent
    agent = init_agent(args, env, device=device)

    # =========
    # Start training
    print(f'Starting training, {args.num_episode} episodes')
    for episode_idx in range(args.num_episode):
        # ==
        # Reset environment and agent
        observation = env.reset()
        action = agent.begin_episode(observation)

        # Counters
        cumu_reward = 0.0
        timestep = 0

        # ==
        # (optional) Record video
        video = None
        max_vid_len = 200
        if args.video_freq is not None:
            if episode_idx % int(args.video_freq) == 0:
                # Render first frame and insert to video array
                frame = env.render()
                video = np.zeros(shape=((max_vid_len,) + frame.shape),
                                 dtype=np.uint8)  # (max_vid_len, C, W, H)
                video[0] = frame

        # ==
        # Run episode
        while True:
            # ==
            # Interact with environment
            observation, reward, done, info = env.step(action)
            action = agent.step(observation, reward, done)

            # ==
            # Counters
            cumu_reward += reward
            timestep += 1

            # ==
            # Optional video
            if video is not None:
                if timestep < max_vid_len:
                    video[timestep] = env.render()

            # ==
            # Episode done
            if done:
                # Logging
                if args.log_dir is not None:
                    # Add reward
                    logger.add_scalar('Reward', cumu_reward,
                                      global_step=episode_idx)
                    # Optionally add video
                    if video is not None:
                        # Determine last frame
                        last_frame_idx = timestep + 2
                        if last_frame_idx > max_vid_len:
                            last_frame_idx = max_vid_len

                        # Change to tensor
                        vid_tensor = torch.tensor(video[:last_frame_idx, :, :, :],
                                                  dtype=torch.uint8)
                        vid_tensor = vid_tensor.unsqueeze(0)

                        # Add to tensorboard
                        logger.add_video('Run_Video', vid_tensor,
                                         global_step=episode_idx,
                                         fps=8)

                    # Occasional print
                    if episode_idx % 100 == 0:
                        print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                else:
                    print(f'Epis {episode_idx}, Timesteps: {timestep}, Return: {cumu_reward}')

                # Agent logging TODO: not sure if this is the best practice
                agent.report(logger=logger, episode_idx=episode_idx)
                break

            # TODO: have some debugging print-out (e.g. every 100 episode) to make sure times and
            # things are good and training is happening

    env.close()
    if args.log_dir is not None:
        logger.close()


if __name__ == "__main__":

    # =====================================================
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='RL in discrete environment')

    # ===
    # Environmental parameters
    parser.add_argument('--env_name', type=str,
                        default='MiniGrid-Empty-6x6-v0', metavar='N',
                        help='environment to initialize (default: CartPole-v1')
    parser.add_argument('--num_episode', type=int, default=20, metavar='N',
                        help='number of episodes to run the environment for '
                             '(default: 500)')

    # ===
    # Agent config
    parser.add_argument('--agent_type', type=str, default='a2c',
                        help='string of agent type')
    parser.add_argument('--config_path', type=str,
                        default='./dopatorch/agents/a2c/default_config.ini',
                        help='path to the agent configuration .ini file')

    # ==
    # Experimental parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='file path to the log file (default: None, printout instead)')
    parser.add_argument('--video_freq', type=int, default=None, metavar='',
                        help='Freq (in # episodes) to record video, only works'
                             'if log_dir is also provided (default: None)')
    parser.add_argument('--tmpdir', type=str, default='./',
                        help='temporary directory to store dataset for training (default: cwd)')

    # ======
    # Parse arguments
    args = parser.parse_args()
    print(args)

    # =====================================================
    # Initialize GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # =====================================================
    # Set seeds
    torch.cuda.manual_seed_all(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # =====================================================
    # Initialize logging
    if args.log_dir is not None:
        # Tensorboard logger
        logger = SummaryWriter(log_dir=args.log_dir)

        """
        TODO FIX THIS
        Need to add agent parameters along with the env parameters, not sure
        how to incorporate this in a nice way into tensorboard yet
        
        # Add hyperparameters
        logger.add_hparams(hparam_dict=vars(args), metric_dict={})
        """
    else:
        logger = None

    # =====================================================
    # Start environmental interactions
    run_environment(args, device=device, logger=logger)
