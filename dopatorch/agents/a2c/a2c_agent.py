# =============================================================================
# Policy networks
#
# Inspired by and modified from repository:
# https://github.com/ikostrikov/
# Policy network file:
#   pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
# Also the optimization process in file:
#   pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
#
# NOTE
# - not implemented:
#       - Generative Adversarial Imitation Learning (GAIL)
#       - Actor Critic using Kronecker-Factored Trust Region (ACKTR)
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dopatorch.agents.a2c.actor_critic_net import *
from dopatorch.replay_memory.rollout_buffer import RolloutBuffer


class A2CAgent(object):
    def __init__(self,
                 action_space: spaces,
                 observation_shape: Tuple = (20,),
                 observation_dtype: torch.dtype = torch.float,
                 gamma: int = 0.9,
                 use_recurrent_net: bool = False,
                 num_rollout_steps: int = 5,
                 value_loss_coef: float =0.5,
                 entropy_coef: float=0.01,
                 max_grad_norm:float=0.5,
                 use_acktr: bool=False,
                 base=None,  # TODO remove need for base?
                 base_kwargs=None,
                 device='cpu'):

        # TODO define more arguments
        """
        TODO define arguments

        """
        super(A2CAgent, self).__init__()

        # ==
        # Initialize basics
        self.action_space = action_space  # TODO initialize action shape
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.device = device

        self.gamma = gamma
        # Number of steps to consider for recurrence (and rollout buffer)
        self.n_rollout_steps = num_rollout_steps

        # ==
        # Initialize actor-critic net (potential TODO change to net initialization method?)
        self.use_recurrent_net = use_recurrent_net
        self.ac_net = ActorCritic(
            self.observation_shape,
            self.action_space,
            base_kwargs={'recurrent': self.use_recurrent_net})
        self.ac_net.to(self.device)

        # ==
        # Initialize Optimizer
        self.use_acktr = use_acktr
        self.rms_lr = 7e-4  # NOTE from a2c_ppo_acktr/arguments.py
        self.rms_momentum = 0.99  # NOTE from a2c_ppo_acktr/arguments.py
        self.rms_eps = 1e-5  # NOTE from a2c_ppo_acktr/arguments.py
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # Initialize optimization
        if self.use_acktr:
            raise NotImplementedError  # NOTE currently not implemented
        else:
            self.optimizer = optim.RMSprop(self.ac_net.parameters(),
                                           self.rms_lr, eps=self.rms_eps,
                                           alpha=self.rms_momentum)

        # ==
        # Initialize rollout storage buffer
        self.rollout_buffer = RolloutBuffer(
            num_steps=self.n_rollout_steps,
            observation_shape=self.observation_shape,
            observation_dtype=self.observation_dtype,
            action_dim=1,  # TODO change?
            hidden_state_dim=self.ac_net.recurrent_hidden_state_size,
            device=self.device
        )

        # ==
        # Variables
        self.timestep = 0

        self.t_prev_action = None
        self.t_hidden_states = None

    def begin_episode(self, observation: np.ndarray) -> int:
        """
        Start of episode
        :param observation: numpy array observation
        :return: integer index denoting action taken
        """

        # ==
        # Soft-reset the replay buffer
        self.rollout_buffer.reset()

        # ==
        # Save initial observation
        # Cast observation (NOTE assume buffer and model in the same device)
        t_obs = (torch.from_numpy(observation).float()
                 .to(self.device))
        # Save o_{0} to the buffer
        self.rollout_buffer.obs_buffer[0].copy_(t_obs)

        # ==
        # Reset variables
        self.timestep = 0

        # ==
        # Select action
        action_idx = self._select_action()
        self.timestep += 1

        return action_idx

    def step(self, observation: np.ndarray,
             reward: float,
             done: bool) -> int:
        # ==
        # Cast and construct tensors
        # NOTE assume buffer and model in the same device
        t_obs = torch.tensor(observation,
                             dtype=self.observation_dtype).to(self.device)
        t_rew = torch.tensor(reward,
                             dtype=torch.float32).to(self.device)
        t_don = (1.0 - torch.tensor(done, dtype=torch.float32)
                 ).to(self.device)  # 0 for done, 1 for not done
        t_true_termin = (torch.ones((1,), dtype=torch.float32)
                         ).to(self.device)  # for now always 1 (true termination)

        # ==
        # Record trajectory in buffer
        self.rollout_buffer.insert(observation=t_obs,
                                   hidden_state=self.t_hidden_states,
                                   reward=t_rew, action=self.t_prev_action,
                                   done=t_don, true_terminal=t_true_termin)

        # ==
        # Periodic model update (whenever buffer is full or when done)
        if done or (self.rollout_buffer.step >= self.n_rollout_steps):
            self._optimize_model()

            # If not done, cycle the buffer to bring the last entry
            # to the beginning for a new partial trajectory
            # NOTE: if agent takes more action after Done the buffer will
            #       likely break
            if not done:
                self.rollout_buffer.cycle()

        # ==
        # Take action
        action_idx = self._select_action()
        self.timestep += 1
        return action_idx

    def _select_action(self) -> int:
        """
        Select action
        :return: int action index
        """
        # NOTE TODO maybe do the below, but needs self.timestep % num_timesteps
        # assert self.timestep == self.rollout_buffer.step

        # ==
        # Call actor critic net
        # Input: o_t, h_t, mask
        # Output: v, policy parameters, h_{t+1}
        with torch.no_grad():
            # Get o_{t}, h_{t-1}, done_{t}
            # NOTE assumes the model and memory buffer is in the same device
            latest_tuple = self.rollout_buffer.get_latest_states()
            o_t, h_tm1, m_t = latest_tuple

            # Get action-relevant quantities
            t_value, t_actor_features, rnn_hxs = self.ac_net.base(
                inputs=o_t,
                rnn_hxs=h_tm1,
                masks=m_t
            )
            # Action distribution, p(a)
            dist = self.ac_net.dist(t_actor_features)

            # ==
            # Sample action
            # NOTE: use dist.mode() to greedily pick action
            t_action = dist.sample()
            t_action_log_probs = dist.log_probs(t_action)  # action prob TODO NOTE not used
            t_policy_entropy = dist.entropy().mean()  # entropy of actions TODO NOTE not used

        # ==
        # Update tracking variables
        self.t_prev_action = t_action  # TODO do I need to clone or detach? (for all 2)
        self.t_hidden_states = rnn_hxs

        # ==
        # Return action
        # NOTE TODO assumes discrete action for now, need to expand upon
        # this to make it compatible with continuous action spaces
        action = int(t_action.cpu().item())
        return action

    def _optimize_model(self):
        # ==
        # Get the full trajectory, t = 0, ..., T-1
        traj_tuple = self.rollout_buffer.get_trajectory()
        o_traj, h_init, d_traj, a_traj = traj_tuple

        # Run full trajectory through actor-critic network
        traj_Vs, traj_actor_features, t_rnn_hxs = self.ac_net.base(
            inputs=o_traj,
            rnn_hxs=h_init,
            masks=d_traj
        )

        # ==
        # Get action probabilities
        traj_dist = self.ac_net.dist(traj_actor_features)

        a_traj_logprob = traj_dist.log_probs(a_traj)  # size (T, 1)
        pol_entropy = traj_dist.entropy().mean()  # scalar

        # ==
        # Compute the trajectory returns at each step

        # Compute the final step value
        with torch.no_grad():
            # Get o_{T}, h_{T-1}, done_{T}
            latest_tuple = self.rollout_buffer.get_latest_states()
            o_t, h_tm1, m_t = latest_tuple

            # Get final step value
            t_fin_value, __, __ = self.ac_net.base(o_t, h_tm1, m_t)
            t_fin_value = t_fin_value.detach()  # size (1,1)

        # Compute the return at each timestep, shaped (T, 1)
        traj_Gs = self.rollout_buffer.compute_returns(next_value=t_fin_value,
                                                      gamma=self.gamma,
                                                      use_gae=False).detach()

        # ==
        # Optimization
        # potential TODO: implement ACKTR?

        # Compute losses
        advantages = traj_Gs[:-1] - traj_Vs  # size (T, 1)
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * a_traj_logprob).mean()

        # ACKTR
        if self.use_acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # future TODO implement acktr from a2c_acktr.py
            raise NotImplementedError

        # Total loss
        total_loss = ((value_loss * self.value_loss_coef)
                      + action_loss
                      - (pol_entropy * self.entropy_coef))

        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.use_acktr:
            raise NotImplementedError
        else:
            nn.utils.clip_grad_norm_(self.ac_net.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        # TODO return loss or log it somewhere


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = A2CAgent()
    print(agent)
