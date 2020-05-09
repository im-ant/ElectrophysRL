# =============================================================================
# The DQN agent with flat input
#
# Adopt (and simplified for flat input) from the dqn_agent.py code
#
# Author: Anthony G. Chen
# =============================================================================

from collections import deque
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dopatorch.discrete_domains import network
from dopatorch.replay_memory import circular_flat_replay_buffer


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon_final):
    """
    TODO: organize stuff here and add start epsilon argument
    Code is copied largely directly from the Google Dopamine code

    :param decay_period: float, the period over which epsilon is decayed.
    :param step: int, the number of training steps completed so far.
    :param warmup_steps: number of steps taken before epsilon is decayed
    :param epsilon: the final epsilon value
    :return: current epsilon value
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon_final) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon_final)
    return epsilon_final + bonus


class DQNFlatAgent(object):
    """
    The DQN Agent with Flat input
    """

    def __init__(self, num_actions: int,
                 observation_shape: Tuple = (84, ),
                 observation_dtype: torch.dtype = torch.uint8,
                 history_size: int = 4,
                 gamma: int = 0.9,
                 min_replay_history: int = 20000,
                 update_period: int = 4,
                 target_update_period: int = 8000,
                 epsilon_fn=linearly_decaying_epsilon,
                 epsilon_final: float = 0.1,
                 epsilon_decay_period: int = 250000,
                 memory_buffer_capacity: int = 1000000,
                 q_minibatch_size: int = 32,
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Initialize the DQN agent

        TODO for future:
            - Add: optimizer parameters, network parameters

        :param num_actions: number of actions the agent can take at any state.
        :param observation_shape: tuple of ints describing the observation
                                  shape, expects tuple of (n, )
        :param observation_dtype: NOTE: type of observaiton
        :param history_size: int, number of observations to use in state stack.
        :param gamma: decay constant
        :param min_replay_history: number of transitions that should be
            experienced before the agent begins training its value function
        :param update_period: int, number of actions between network training
        :param target_update_period: update period of target network (per action)
        :param epsilon_fn: epsilon decay function
        :param epsilon_start: exploration rate at start
        :param epsilon_final: final exploration rate
        :param epsilon_decay_period: length of the epsilon decay schedule
        :param memory_buffer_capacity: total capacity of the memory buffer for replay
        :param device: 'cuda' or 'cpu', depending on if 'cuda' is available
        :param summary_writer: TODO implement this with TensorBoard in the future
        """

        # ==
        # Definable attributes
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.history_size = history_size
        self.gamma = gamma
        self.min_replay_history = min_replay_history
        self.update_period = update_period
        self.target_update_period = target_update_period
        self.epsilon_fn = epsilon_fn
        self.epsilon_final = epsilon_final
        self.epsilon_decay_period = epsilon_decay_period

        self.memory_buffer_capacity = memory_buffer_capacity
        self.minibatch_size = q_minibatch_size
        self.seed = seed
        self.device = device

        self.rng = np.random.RandomState(seed=self.seed)

        # ==
        # Initialize network, memory and optimizer
        self.policy_net = None
        self.target_net = None
        self.memory = None

        self._init_network()
        self._init_memory()

        # https://pytorch.org/docs/stable/optim.html#torch.optim.RMSprop
        # TODO change to Adam?
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),
                                       lr=0.00025,
                                       alpha=0.95,
                                       momentum=0.0,
                                       eps=0.00001,
                                       centered=True)

        # History queue: for stacking observations (np matrices in cpu)
        self.history_queue = deque(maxlen=self.history_size)

        # ==
        # Counter variables
        self.total_actions_taken = 0  # for epsilon decay
        self.total_optim_steps = 0  # for target network updates
        self._latest_epsilon = 1.0
        self.per_episode_log = {
            't': 0,
            'Q_optim_steps': 0,
            'total_Q_loss': 0.0,
        }

        self.action = None  # action to be taken (selected at prev timestep)
        self._prev_observation = None

    def _init_network(self) -> None:
        """Initialize the small flat Q network"""
        # Take product of input shape to handle both (n,) and (1, n)
        self.policy_net = network.mlp_network(
            input_size=(np.prod(self.observation_shape)*self.history_size),
            hidden_size=32,
            num_actions=self.num_actions
        ).to(self.device)
        self.target_net = network.mlp_network(
            input_size=(np.prod(self.observation_shape)*self.history_size),
            hidden_size=32,
            num_actions=self.num_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # set target net to evaluation mode

    def _init_memory(self) -> None:
        """Initialize the memory buffer"""
        self.memory = circular_flat_replay_buffer.CircularFlatReplayBuffer(
            buffer_cap=self.memory_buffer_capacity,
            history=self.history_size,
            obs_shape=self.observation_shape,
            obs_dtype=self.observation_dtype,
            device=self.device
        )

    def begin_episode(self, observation: np.ndarray) -> int:
        """
        Start the episode
        :param observation: first observation, with env.observation_space.shape
        :return: first action (idx) to be taken
        """
        # Initialize (zero-padded) history
        # NOTE: each entry in queue is a (self.observation_shape) tensor
        for _ in range(self.history_size):
            zero_pad_mat = torch.zeros(self.observation_shape,
                                       dtype=self.observation_dtype,
                                       device='cpu')
            self.history_queue.append(zero_pad_mat)
        # Add the first observation to the history
        obs_tensor = torch.tensor(observation,
                                  dtype=self.observation_dtype,
                                  device='cpu')
        self.history_queue.append(obs_tensor)

        # ==
        # Reset per-episode counters
        for k in self.per_episode_log:
            self.per_episode_log[k] *= 0

        # ==
        # Select action update a_0, o_0
        self.action = self._select_action()
        self._prev_observation = observation
        self.total_actions_taken += 1
        # TODO potential bug above that might skip some training steps?
        return self.action

    def step(self, observation: np.ndarray, reward: float, done: bool) -> int:
        """
        The agent takes one step

        :param observation: o_t, observation from environment, should
                            have shape self.observation_shape
        :param reward: r_t, float reward received
        :param done: done_t, bool, whether the episode is finished
        :return: int denoting action to take at this step
        """

        # ==
        # Construct state with new observation, o_t
        cur_obs_tensor = torch.tensor(observation,
                                      dtype=self.observation_dtype,
                                      device='cpu')
        self.history_queue.append(cur_obs_tensor)

        # ==
        # Store experience: o_{t-1}, a_{t-1}, r_t, done_t
        # First set them to the correct tensor, dtype and device
        obs_tensor = torch.tensor(self._prev_observation,
                                  dtype=self.observation_dtype,
                                  device=self.device)
        act_tensor = torch.tensor([self.action], dtype=torch.int32,
                                  device=self.device)
        rew_tensor = torch.tensor([reward], dtype=torch.float32,
                                  device=self.device)
        don_tensor = torch.tensor([done], dtype=torch.bool,
                                  device=self.device)
        self.memory.push(obs_tensor, act_tensor, rew_tensor, don_tensor)

        # ==
        # Training step
        self._train_step()
        self.total_actions_taken += 1
        self.per_episode_log['t'] += 1

        # ==
        # Select action and update a_{t-1}, o_{t-1}
        self.action = self._select_action()
        self._prev_observation = observation
        return self.action

    def _select_action(self) -> int:
        """
        Select action according to the epsilon greedy policy
        :return: int denoting action to take
        """

        # Compute epsilon
        epsilon = self.epsilon_fn(self.epsilon_decay_period,
                                  self.total_actions_taken,
                                  self.min_replay_history,
                                  self.epsilon_final)
        self._latest_epsilon = epsilon

        # ===
        # Epsilon greedy policy
        if random.random() <= epsilon:
            # random action with probability epsilon
            return self.rng.choice(self.num_actions)
        else:
            # Construct state
            state = self._history_queue_to_state()

            # greedy action
            with torch.no_grad():
                # Get values (1, n_actions), then take max column index
                action_tensor = self.policy_net(state).max(1)[1].view(1, 1)
            return action_tensor.item()

    def _history_queue_to_state(self) -> torch.tensor:
        """
        Convert the current history queue into a torch tensor state, where the
        state is sent to device (same as the neural nets).

        Each entry in the self.history_queue is a (self.observation_shape)
        tensor, which is concatenated at the first dimension, with an
        additional dimension added at the 0-oth position

        :return: 4-dimensional torch.tensor object of state, consisting of
                dims: (1, self.observation_shape[0], ...)
        """
        state = torch.cat(list(self.history_queue), dim=0).unsqueeze(0)
        state = state.type(torch.float32).to(self.device)
        return state

    def _train_step(self):
        """Runs a single training step.
        Runs a training op if both:
          (1) A minimum number of frames have been added to the replay buffer.
          (2) `training_steps` is a multiple of `update_period`.
        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if len(self.memory) > self.min_replay_history:
            if self.total_actions_taken % self.update_period == 0:
                if len(self.memory) >= self.minibatch_size:
                    self._optimize_model()
                # TODO: Dopamine had a bunch of summary writers here
            # Update target network
            if self.total_optim_steps % self.target_update_period == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def _optimize_model(self) -> float:
        """
        Optimizes the policy (Q) network
        :return:
        """
        # If not enough memory
        if len(self.memory) < self.minibatch_size:
            return -1.0

        # ==
        # Sample memory and unpack
        mem_batch = self.memory.sample(self.minibatch_size)
        (state_batch, action_batch, reward_batch,
         next_state_batch, done_batch) = mem_batch

        state_batch = state_batch.type(torch.float32).to(self.device)
        action_batch = action_batch.type(torch.long).to(self.device)
        reward_batch = reward_batch.type(torch.float32).to(self.device)
        next_state_batch = next_state_batch.type(torch.float32).to(self.device)
        done_batch = done_batch.type(torch.bool).to(self.device)

        # ==
        # Compute TD error

        # Get policy net output (batch, n_actions), extract action (index)
        # which need to have shape (batch, 1) for torch.gather to work.
        state_action_values = self.policy_net(state_batch) \
            .gather(1, action_batch)  # (batch-size, 1)

        # Get semi-gradient Q-learning targets (no grad to next state)
        next_state_values = self.target_net(next_state_batch) \
            .max(1)[0].unsqueeze(1).detach()  # (batch-size, 1)
        # Note that if episode is done do not use bootstrap estimate
        expected_state_action_values = (((next_state_values * (~done_batch))
                                         * self.gamma)
                                        + reward_batch)

        # Compute TD loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # ==
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # gradient-clipping
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Log the loss
        if loss is not None:  # TODO: should never come to this?
            self.per_episode_log['total_Q_loss'] += loss.item()
        self.per_episode_log['Q_optim_steps'] += 1
        self.total_optim_steps += 1

    def report(self, episode_idx, logger=None):
        if episode_idx == 0:
            print("\tEpsilon || total_actions || total_optims || avg_Q_loss")

        # ==
        # Compute the averages
        avg_Q_loss = 0.0
        if self.per_episode_log['Q_optim_steps'] > 0:
            avg_Q_loss = (self.per_episode_log['total_Q_loss'] /
                          self.per_episode_log['Q_optim_steps'])

        # ==
        # Print or log
        if logger is None:
            print(f"  {self._latest_epsilon} || "
                  f"{self.total_actions_taken} || "
                  f"{self.total_optim_steps} || "
                  f"{avg_Q_loss}")
        else:
            logger.add_scalar('Timesteps', self.per_episode_log['t'],
                              global_step=episode_idx)
            logger.add_scalar('Eps_exploration', self._latest_epsilon,
                              global_step=episode_idx)
            logger.add_scalar('Total_actions', self.total_actions_taken,
                              global_step=episode_idx)
            logger.add_scalar('Total_Q_optimizations', self.total_optim_steps,
                              global_step=episode_idx)
            logger.add_scalar('Epis_Q_loss', avg_Q_loss,
                              global_step=episode_idx)


if __name__ == "__main__":
    # for testing run this directly
    print('testing')
    agent = DQNFlatAgent(num_actions=8)
    print(agent)
    print(agent.policy_net)
    print(agent.target_net)
    print(agent.memory.capacity)
