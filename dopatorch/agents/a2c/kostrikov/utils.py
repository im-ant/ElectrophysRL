# =============================================================================
# Utility helper methods
#
# Copied almost directly from with slight refactoring:
# Repo: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/
# File: blob/master/a2c_ppo_acktr/utils.py
#
# Methods removed:
#   - get_render_func(venv)
#   - get_vec_normalize(venv)
#
# Original author repository: ikostrikov
# =============================================================================

import glob
import os

import torch
import torch.nn as nn


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
