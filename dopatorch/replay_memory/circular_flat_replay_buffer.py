# ============================================================================
# Module implementing a replay memory buffer where the observation is
# assumed to be flat - i.e. have shape (n, ).
#
# The reason a specific "flat" replay buffer is required is because it changes
# the sampling and observation aggregation slightly from the convolution memory
# version.
#
# Copied from the conv DQN implementation with modifications for the
# flat-input setting
# https://github.com/im-ant/gym_RL/blob/control-v1/MiniGrid/replay_memory.py
#
# Author: Anthony G. Chen
# ============================================================================

from typing import List

import numpy as np
import torch


class CircularFlatReplayBuffer(object):
    """
    Circular replay buffer for naive uniform sampling of the recent past
    """

    def __init__(self, buffer_cap=50000,
                 history=1,
                 obs_shape=(100,),
                 obs_dtype=torch.float32,
                 seed=0,
                 device='cpu') -> None:
        """
        Initializing the circular replay buffer. For a flat setting, the
        internal buffer should have shapes:
            _obs_buffer : (capacity, 147)  or (capacity, obs_shape[0])
            _act_buffer : (capacity, 1)
            _rew_buffer : (capacity, 1)
            _done_buffer: (capacity, 1)

        :param buffer_cap: total capacity of the buffer
        :param history: number of observation to stack to make a state
        :param obs_shape: shape of observation tensor
        :param device: device to store memory on
        """

        # Initialize counter
        self._cur_idx = 0  # Current buffer index to write to
        self.size = 0  # Number of experiences stored
        self.capacity = buffer_cap  # Total buffer capacity
        self.history = history  # History length (# frames for a state)
        self._device = device  # Device (cpu/cuda) to store buffers on

        # Initialize the experience shapes and types
        self._obs_shape = obs_shape
        self._obs_dtype = obs_dtype
        self._act_dtype = torch.int32
        self._rew_dtype = torch.float32

        # Initialize the experience buffers
        obs_buffer_shape = ((self.capacity,) + self._obs_shape)
        self._obs_buffer = torch.empty(obs_buffer_shape, dtype=self._obs_dtype,
                                       device=self._device)
        self._act_buffer = torch.empty((self.capacity, 1), dtype=self._act_dtype,
                                       device=self._device)
        self._rew_buffer = torch.empty((self.capacity, 1), dtype=self._rew_dtype,
                                       device=self._device)
        self._done_buffer = torch.empty((self.capacity, 1), dtype=torch.bool,
                                        device=self._device)

        # Rng for sampling
        self.rng = np.random.RandomState(seed=seed)

    def push(self, observ: torch.tensor, action: torch.tensor,
             reward: torch.tensor, done: torch.tensor) -> None:
        """
        Pushes an experience to the buffer.
        Note we assume each experience is a tuple containing items:
            (o_{t-1}, a_{t-1}, r_t, done_t)
        """
        # Write experiences to buffer
        # TODO: add assertions to check for types?
        self._obs_buffer[self._cur_idx] = observ
        self._act_buffer[self._cur_idx] = action
        self._rew_buffer[self._cur_idx] = reward
        self._done_buffer[self._cur_idx] = done

        self._cur_idx = (self._cur_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) \
            -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """
        Sample minibatch experience from the buffer
        :param n: batch size
        :return: Minibatch memory of the following shape:
                     state_batch: (batch size, history*self.obs_shape[0])
                     next_state_batch: same as state_batch
                     act_batch: (batch size, 1)
                     rew_batch: (batch size, 1)
                     don_batch: (batch size, 1)
        """
        # Sample indeces, (inclusive) last indeces of the sequences
        # TODO: modify such one cannot sample index i where done[i-1]==True
        #   this is because if i is the last obs of successor state then the
        #   previous state is completely absent
        indeces = self.rng.choice(self.size, size=n, replace=False)

        # Get the buffered experiences
        state_batch, next_state_batch = self.encode_states(indeces)
        act_batch = self._act_buffer[indeces]
        rew_batch = self._rew_buffer[indeces]
        don_batch = self._done_buffer[indeces]

        return state_batch, act_batch, rew_batch, next_state_batch, don_batch

    def encode_states(self, idxs: np.ndarray) -> (torch.tensor, torch.tensor):
        """
        Generate the state (stacks of observations) given the indeces of the
        last observation, zero-padding if crossing episode boundaries
        :param idxs: indeces of the last observation of the successor state
        :return: minibatches of state and successor states: cur_states,
                 nex_states, both of shape (batch_size, feature)
        """

        # Initialize current and next states tensors (pre zero-padded)
        # State shape: (batch size, history*obs_shape[0])
        _state_tensor_shape = (len(idxs), self.history * self._obs_shape[0])
        cur_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)
        nex_states = torch.zeros(_state_tensor_shape,
                                 dtype=self._obs_dtype,
                                 device=self._device)

        # Fill each state
        for i, buf_idx in enumerate(idxs):
            # Get the idxs of valid observation sequence of len history + 1
            # TODO maybe fix this indexing issue, should be just (buf_idx) not (buf_idx+1)?
            # not sure if this is an actual issue yet thought
            seq_idxs = self._get_valid_seq((buf_idx + 1) % self.size)
            cur_seq_idxs = seq_idxs[:-1]
            nex_seq_idxs = seq_idxs[1:] if len(seq_idxs) > self.history \
                else seq_idxs

            # Construct state using the observation seq indeces
            # NOTE reshaping stacks multiple channels (if present) AND frames
            #      into the 2nd dimension of the returned state tensor
            if len(cur_seq_idxs) > 0:
                # number of (non zero-padded) observations & features to stack
                non_zero_len = len(cur_seq_idxs) * self._obs_shape[0]
                # note: reshape each state from
                #       ( len(seq_idxs), feature )
                #       to (len(seq_idxs)*feature, )
                cur_states[i, -non_zero_len:] = torch.reshape(
                    self._obs_buffer[cur_seq_idxs],
                    (non_zero_len,)
                )

            if len(nex_seq_idxs) > 0:
                non_zero_len = len(nex_seq_idxs) * self._obs_shape[0]
                nex_states[i, -non_zero_len:] = torch.reshape(
                    self._obs_buffer[nex_seq_idxs],
                    (non_zero_len,)
                )

        return cur_states, nex_states

    def _get_valid_seq(self, last_idx: int) -> List[int]:
        """
        Helper method to return a sequence of valid (observation) indeces
        (inclusive) to form (both) the previous and successor states.
        Length could range from be 1 to self.history+1

        :param last_idx: last index of the sequence
        """
        assert last_idx < self.size

        # Get the allowable first index of this sequence
        first_idx = last_idx
        for j in range(1, self.history + 1):
            cur_idx = last_idx - j
            # If it goes out of bound, determine how to treat
            if cur_idx < 0:
                if self.size == self.capacity:
                    cur_idx = cur_idx % self.capacity
                else:
                    break
            # If it crosses an episode boundary
            if self._done_buffer[cur_idx]:
                break
            # If all good then keep going
            first_idx = cur_idx

        # Get the sequence of indeces
        valid_seq = []
        while first_idx != ((last_idx + 1) % self.capacity):
            valid_seq.append(first_idx)
            first_idx = (first_idx + 1) % self.capacity

        return valid_seq

    def __len__(self) -> int:
        return self.size


def main():
    """For testing / debug purposes only """
    buf = CircularFlatReplayBuffer(buffer_cap=15,
                                   obs_shape=(2,),
                                   history=3
                                   )

    for _ob in range(20):
        cur_obs = torch.tensor([1.0, 1.0]) * (_ob + 1)
        is_done = False

        if (_ob + 1) % 8 == 0:
            is_done = True

        buf.push(cur_obs, 1, 2, is_done)

        print(len(buf), buf.capacity)

    print(buf._done_buffer)
    # print(buf.get_valid_seq(9))

    # print(buf.encode_states([1, 8]))

    s, a, r, sp, d = buf.sample(3)
    print(s)
    print('===')
    print(sp)
    print(a)
    print(r)


if __name__ == "__main__":
    main()
