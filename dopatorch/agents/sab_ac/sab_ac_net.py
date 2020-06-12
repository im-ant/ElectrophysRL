# =============================================================================
# Actor-Critic networks
#
# Inspired by and adapted from the policy networks in:
# Repo: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
# File: blob/master/a2c_ppo_acktr/model.py
#
# Original author repository: ikostrikov
# Adapted by: Anthony G. Chen
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dopatorch.agents.a2c.kostrikov.distributions import (Bernoulli,
                                                          Categorical,
                                                          DiagGaussian)
from dopatorch.agents.a2c.kostrikov.utils import init
import dopatorch.agents.sab_ac.mem.SAB.sab_nn as sab_nn


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(ActorCritic, self).__init__()
        # ==
        # Initialize the base network

        # Set up base class
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        # Initialize base
        # TODO should just give the full obs shape (tuple)
        self.base = base(obs_shape[0], **base_kwargs)

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

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.rnn = sab_nn.SAB_LSTM(input_size=recurrent_input_size,
                                       hidden_size=hidden_size,
                                       output_size=hidden_size,
                                       num_layers=1,
                                       truncate_length=50,
                                       remem_every_k=1,
                                       k_top_attn=5,
                                       block_attn_grad_past=False)

            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        # TODO change to just hidden state size and return the last layer instead
        # or do this separately?
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        # TODO why is it named output_size
        return self._hidden_size

    def _forward_rnn(self, x, hxs, mem, masks):
        """
        Forward pass in recurrent model. Assumes data is never in batch mode
        (since actor-critic is an on-policy method). During a typical agent
        "step", T=1. During actual update, T=sequence length to learn over.

        Note: in the old code from ikostrikov/pytorch-a2c-ppo-acktr-gail,
              we had to first divide up each episode by the done mask, then
              process each episode together as a batch. Here, we assume that
              the sequence _never_ cross episode boundaries, so we simplify
              _always_ processing the sequence as a single RNN batch

        :param x: input / input sequence, shape (T, *obs_shape)
                  more specifically, *obs_shape _should_ be flat (i.e. if it
                  was an image it should have been embedded first)
        :param hxs: Tuple of (h_t, c_t)
                        each are shape (1, hidden_dim)
        :param mem: memory tensor
        :param masks: done masks (1 for not done), shape (T, 1)
        :return: TODO write this
        """

        # ==
        # Reshape inputs to include batch dimension (of size 1)
        T = x.size(0)

        x = x.unsqueeze(1)  # (T, 1, *obs_shape)
        # LSTM operation
        hxs = tuple(
            [hx.unsqueeze(1) for hx in hxs]
        )  # Both h_t and c_t are now (1, 1, hidden_dim)

        # TODO modify things here with memory?

        # ==
        # Pass sequence through recurrent net
        y, hxs, mem = self.rnn(x, hxs, mem)

        # ==
        # Reshape to remove batch dimension and return
        y = y.squeeze(1)  # (T, output_dim)
        # LSTM operation, change h_t and c_t back into (1, hidden_dim)
        hxs = tuple([hx.squeeze(1) for hx in hxs])

        return y, hxs, mem


class CNNBase(NNBase):
    """
    NOTE: not quite considered when developing. Probably not going to work.
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):

        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # TODO need to do some calculation here on the dimensions given differently
        # sized inputs
        #
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, mem, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs, mem = self._forward_rnn(x, rnn_hxs, mem, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, mem


if __name__ == "__main__":
    model = sab_nn.SAB_LSTM(input_size=10, hidden_size=32, output_size=10, num_layers=1)
    print(model)

    for name, param in model.named_parameters():
        print(name, param.size())

    pass
