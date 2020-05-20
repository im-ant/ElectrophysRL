# =============================================================================
# Rollout buffer
#
# Inspired by and adapted from the policy networks in:
# repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# File: blob/master/a2c_ppo_acktr/storage.py
#
# Some notes
#   - Remove methods feed_forward_generator and recurrent_generator as they
#       are exclusive to PPO. Potential future inclusion.
#
#
# Original repository author: ikostrikov
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer(object):
    def __init__(self,
                 num_steps: int,
                 observation_shape: Tuple,
                 observation_dtype: torch.dtype = torch.float32,
                 action_dim: int = 1,
                 hidden_state_dim: int = 16,
                 device: str = 'cpu'):
        """
        Roll-out trajectory experience buffer. Anticipated use case is for
        storing on-policy trajectories for on-policy methods (e.g. A2C).
        Also works with recurrent nets by using the full trajectory.

        :param num_steps: max number of steps to keep in the buffer
        :param observation_shape: shape tuple of observation
        :param observation_dtype: type to cast the observation buffer to
        :param action_dim: int, dimension of action output
        :param hidden_state_dim: int, dimension of the hidden state
        :param device: 'cpu' / 'cuda'
        """

        # ==
        # Initialize attributes
        self.num_steps = num_steps
        self.step = 0
        self.device = device

        # Size and types
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.hidden_state_dim = hidden_state_dim

        self._obs_dtype = observation_dtype
        self._hid_dtype = torch.float32
        self._rew_dtype = torch.float32
        self._act_dtype = torch.int32

        # ==
        # Initialize buffers

        # Experience buffers
        self.obs_buffer = torch.zeros((self.num_steps + 1,
                                       *self.observation_shape),
                                      dtype=self._obs_dtype).to(self.device)
        self.hid_buffer = torch.zeros((self.num_steps + 1,
                                       self.hidden_state_dim),
                                      dtype=self._hid_dtype).to(self.device)
        self.rew_buffer = torch.zeros((self.num_steps, 1),
                                      dtype=self._rew_dtype).to(self.device)
        self.act_buffer = torch.zeros((self.num_steps, self.action_dim),
                                      dtype=self._rew_dtype).to(self.device)

        # Termination (done) mask: 0 for done, 1 for not done
        self.don_buffer = torch.ones((self.num_steps + 1, 1),
                                     dtype=torch.float32).to(self.device)
        # True termination buffer. 0 for not true termination (i.e. due to
        # time limit), 1 for true termination
        self.true_termin = torch.ones((self.num_steps + 1, 1),
                                      dtype=torch.float32).to(self.device)

    def to(self, device) -> None:
        """
        Send the buffer to specified device
        DEPRECATED. Not used. Can just initialize with a certain device which
        allows for more consistency and less confusion in the agent code.
        :param device: e.g. 'cpu' or 'cuda'
        :return: None
        """
        self.obs_buffer = self.obs_buffer.to(device)
        self.hid_buffer = self.hid_buffer.to(device)
        self.rew_buffer = self.rew_buffer.to(device)
        self.act_buffer = self.act_buffer.to(device)
        self.don_buffer = self.don_buffer.to(device)
        self.true_termin = self.true_termin.to(device)

        self.device = device

    def insert(self,
               observation: torch.tensor,
               hidden_state: torch.tensor,
               reward: torch.tensor,
               action: torch.tensor,
               done: torch.tensor,
               true_terminal: torch.tensor) -> None:
        """
        Store single experience. Inserted in such a way so that the same
        index across all the buffers should contain:
            o_{t}, h_{t-1}, a_{t}, log_p_a_{t}, hat_V_{t}, r_{t+1},
            done_{t}, true_terminal_{t}

        Where the time index is determined by whether the variable was
        generated before or after the env.step() step

        :param observation: observation tensor, at {t+1}
        :param hidden_state: hidden state tensor, at {t}
        :param reward: reward tensor, at {t+1}
        :param action: action tensor, at {t}
        :param done: done tensor, at {t+1}
        :param true_terminal: true termination state tensor, at {t+1}
        :param pred_value: predicted value tensor, at {t}
        :param action_log_prob: action log probability tensor, at {t}
        :return: None
        """

        # ==
        # Save copy of experiences
        self.obs_buffer[self.step + 1] = observation.clone().detach()
        self.hid_buffer[self.step + 1] = hidden_state.clone().detach()
        self.don_buffer[self.step + 1] = done.clone().detach()
        self.true_termin[self.step + 1] = true_terminal.clone().detach()

        self.rew_buffer[self.step] = reward.clone().detach()
        self.act_buffer[self.step] = action.clone().detach()

        self.step += 1  # potential TODO confirm correctness
        # old: = (self.step + 1) % self.num_steps

    def get_latest_states(self) -> Tuple:
        """
        Get the most recent observation, hidden state and done mask.
        The point of this method is mainly to help with agent action selection
        :return: obs_{t}, shape (1, *self.observation_shape)
                 h_{t-1}, shape (1, hidden_state_dim)
                 done_{t}, shape (1, 1)
        """
        # Ensure index is in an allowable range
        assert self.step <= self.num_steps

        o_t = self.obs_buffer[self.step].unsqueeze(0)
        h_tminus1 = self.hid_buffer[self.step].unsqueeze(0)
        m_t = self.don_buffer[self.step].unsqueeze(0)

        return o_t, h_tminus1, m_t

    def get_trajectory(self) -> Tuple:
        """
        Get the full trajectory up to the current timepoint. A sort of helper
        method with the actor critic agent's optimization step.
        :return: obs_traj, shape (T, *obs_shape)
                 h_{0}, shape (1, hidden_dim)
                 done_traj, shape (T, 1)
                 action_traj, shape (T, action_dim)
        """
        # Ensure index is in an allowable range
        assert self.step <= self.num_steps
        # TODO NOTE consider edge case: what happens if this is called right
        # after self.cycle() has been called? does it still work?

        # ==
        # Get the trajectory up to current timestep
        o_traj = (self.obs_buffer[:self.step]
                  .view(-1, *self.observation_shape))  # (T, *obs_shape)
        h_init = (self.hid_buffer[0]
                  .view(-1, self.hidden_state_dim))  # (1, hidden_dim)
        d_traj = (self.don_buffer[:self.step]
                  .view(-1, 1))  # (T, 1)
        a_traj = (self.act_buffer[:self.step]
                  .view(-1, self.action_dim))  # (T, action_dim)

        # Return
        return o_traj, h_init, d_traj, a_traj

    def reset(self, hard=False) -> None:
        """
        Reset the trajectory buffer for a new sequence of experience.
        Default is a "soft" reset, which just resets the values at the
            0th index (i.e. t=0) since future values will be overwritten
            by the insert method anyway.

        :param hard: re-initialize the entire buffer
        :return: None
        """

        if hard:
            raise NotImplementedError

        self.obs_buffer[0] = self.obs_buffer[0] * 0
        self.hid_buffer[0] = self.hid_buffer[0] * 0
        self.rew_buffer[0] = self.rew_buffer[0] * 0
        self.act_buffer[0] = self.act_buffer[0] * 0

        self.don_buffer[0] = self.don_buffer[0] * 0 + 1
        self.true_termin[0] = self.true_termin[0] * 0 + 1

        self.step = 0

    def cycle(self) -> None:
        """
        Cycle the buffer to get the latest states to the beginning
        :return: None
        """
        # Cycle the observation and hidden state to continue episode
        self.obs_buffer[0].copy_(self.obs_buffer[-1])
        self.hid_buffer[0].copy_(self.hid_buffer[-1])
        # Cycle done-ness (NOTE: it should always be NOT done, but here
        # it is implemented just for consistency)
        self.don_buffer[0].copy_(self.don_buffer[-1])
        self.true_termin[0].copy_(self.true_termin[-1])

        self.step = 0

    def compute_returns(self,
                        next_value: torch.tensor,
                        gamma: float,
                        use_gae: bool = False,
                        gae_lambda: float = 0.95,
                        traj_pred_values: torch.tensor = None,
                        use_proper_time_limits: bool = False) -> torch.tensor:
        """
        Compute the per-step returns from 0 to self.step (denoted as T below)

        :param next_value: tensor of shape (1,1) denoting the predicted value
                           at V_{T}. Used for bootstrapping
        :param gamma: discount factor
        :param use_gae: Use generalized advantage estimator (GAE)
        :param gae_lambda: lambda parameter for GAE
        :param traj_pred_values: full trajectory of predicted values, needed
                                 if using GAE. Shape (T, 1)
        :param use_proper_time_limits: (not implemented), whether to
                                 distinguish between timeout done or real done
        :return: tensor of per-step returns, shape (T+1, 1)
        """
        # ==
        # Initialize the returns tensor
        t_returns = torch.zeros((self.step + 1, 1),
                                dtype=torch.float32)

        # ==
        # Compute return
        if use_proper_time_limits:
            # future TODO implement this, referencing below and
            # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
            # blob/master/a2c_ppo_acktr/storage.py
            raise NotImplementedError
        else:
            if use_gae:
                # NOTE TODO this has not been validated to work yet.
                # Initialize predicted values
                t_pred_Vs = torch.zeros((self.step + 1, 1),
                                        dtype=torch.float32)
                t_pred_Vs[0:self.step] = traj_pred_values.clone().detach()
                t_pred_Vs[-1] = next_value

                # Compute GAE returns
                gae = 0
                for t in reversed(range(self.step)):
                    delta = (self.rew_buffer[t]
                             + (gamma * t_pred_Vs[t + 1]
                                * self.don_buffer[t + 1])
                             - t_pred_Vs[t])
                    gae = (delta + (gamma * gae_lambda
                                    * self.don_buffer[t + 1] * gae))
                    t_returns[t] = gae + t_pred_Vs[t]
            else:
                # Set (predicted) G_{T}
                t_returns[-1] = next_value
                # Iterate from t = T-1, T-2, ..., 0
                for t in reversed(range(self.step)):
                    t_returns[t] = ((t_returns[t + 1]
                                     * gamma * self.don_buffer[t + 1])
                                    + self.rew_buffer[t])

        return t_returns
