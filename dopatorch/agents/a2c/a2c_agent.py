# =============================================================================
# Policy networks
#
# Inspired by and modified from the policy networks in:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
# as well as the optimization process in:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
#
# Author: Anthony G. Chen
# =============================================================================

import numpy as np
import torch

from dopatorch.agents.a2c.policy_net import *


class A2CAgent(object):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        """

        TODO define arguments
        :param obs_shape:
        :param action_space:
        :param base:
        :param base_kwargs:
        """
        super(A2CAgent, self).__init__()

        # ==
        # Initialize the actor-critic policy network
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            # future TODO: implement CNNBase in (line 21-22):
            # pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)


        """ TODO change this to be something else?
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        """

        """
        NOTE: "agent" is a combination of:
            https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/main.py
            https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
            https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/a2c_acktr.py
            
        TODO
        - use the RolloutStorage as-is
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py
        - just get the code to work, via a combination of the training steps from main and model.py
        
        """

        self.rnn_hxs = None     # store RNN hidden states


    def begin_episode(self) -> None:
        pass

    def step(self, observation: np.ndarray) -> int:
        # =
        # Cast
        obs_tensor = torch.tensor(observation)

        # ==
        # Sample and return action

        # TODO below is incorrect, need to define the rollout storage before I can
        # update the below
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self._act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])





    # @property
    def is_recurrent(self):
        return self.base.is_recurrent

    # @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
