# ============================================================================
# Wrapper for MiniGrid enviroment to produce flat output
# Copied from and improved upon:
# https://github.com/im-ant/RL-767/blob/master/flat_SMiRL/envs/flat_wrapper.py
#
# Some reference on MiniGrid
#   How MiniGrid sets up its observation_space:
#       https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py#L677
#
#   How objects are compactly encoded:
#       https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/minigrid.py#L110
#       Looks like objects are encoded using a 0-10 indexing system
#
# Author: Anthony G. Chen
# ============================================================================

import warnings

import gym
from gym import spaces
from gym import ObservationWrapper
import numpy as np


class MiniGridFlatWrapper(gym.core.Wrapper):
    def __init__(self, env,
                 use_tensor=True,
                 scale_observation=True,
                 scale_min=0.0,
                 scale_max=10.0):
        """
        Initializes the environment wrapper
        :param env: assumed to be a raw MiniGrid environment without
                    additional wrappers
        :param use_tensor: whether use a flat (n,) vector observation, or
                           a (1, n) tensor observation. Using tensor is
                           crucial when using deep NN and memory replay
        :param scale_observation: whether to scale the observation to have
                                  range [0,1]
        :param scale_low: if scale, set lowerbound on scale range
        :param scale_high: if scale, set upperbound on scale range
        """
        super(MiniGridFlatWrapper, self).__init__(env)

        # Whether to output (1, n) tensor as supposed to flat (n,) vector
        self.use_tensor = use_tensor

        # Set-up whether to scale the observation range and if so how much
        self.scale_observation = scale_observation
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        # Initialize the wrapper observation space
        self.observation_space = self._init_observation_space()

        # TODO set seed

    def _init_observation_space(self):
        """
        Helper method to initialize a flat observation space
        :return: gym.spaces.Box with the right observation space
        """
        # Get the observation space from the raw environment
        # NOTE assumes a raw MiniGrid space which is a dictionary whose
        # 'image' entry contains the actual image-encoding observation
        raw_obs_space = self.env.observation_space['image']

        # Determine the range and type of observations
        # TODO optional: scale different dimensions differently, which might
        #      be helpful if I want to include things like directionality
        # TODO scaling different is super important for the control tasks;
        #      in that case I might want to set a absolute upper and lower
        #      bound, and use the env ranges (clipped at absolute bounds)
        #      as the actual range
        if self.scale_observation:
            obs_low = 0.0
            obs_high = 1.0
            obs_dtype = np.dtype(np.float32)
        else:
            # NOTE: assume all dimensions have the same range
            obs_low = np.min(raw_obs_space.low)
            obs_high = np.max(raw_obs_space.high)
            obs_dtype = raw_obs_space.dtype

        # Set up observation space shape
        if self.use_tensor:
            obs_shape = (1, np.prod(raw_obs_space.shape))
        else:
            obs_shape = (np.prod(raw_obs_space.shape),)

        # Set up observation space
        new_obs_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=obs_dtype
        )

        return new_obs_space

    def _transform_observation(self, obs):
        """
        Helper method to transform the raw observation via flattening
        and scaling to [0,1] if required
        :param obs: raw observation from MiniGrid
        :return: processed observation
        """

        # ==
        # Flatten
        obs = obs['image']
        obs = np.ravel(obs)

        # ==
        # Scale range
        if self.scale_observation:
            # Check if value is within bound
            if np.min(obs) < self.scale_min or np.max(obs) > self.scale_max:
                # maybe TODO: warn only once if repeated warning?
                warnings.warn(f'MiniGrid observation value outside scaling'
                              f'range. Observation range: '
                              f'[{np.min(obs)},{np.max(obs)}], Scaling '
                              f'range:[{self.scale_min}, {self.scale_max}].'
                              f' Will automatically clip out-of-range values.',
                              RuntimeWarning)

            # Clip out-of-range values to range
            obs = np.clip(obs, self.scale_min, self.scale_max)

            # Scale values to [0,1]
            obs -= self.scale_min
            obs *= (1.0 / (self.scale_max - self.scale_min))

        # ==
        # Expand to tensor
        if self.use_tensor:
            obs = np.expand_dims(obs, axis=0)

        return obs

    def reset(self, **kwargs):
        """
        Wraps around the reset function to change observation
        :param kwargs: any input to the reset function
        :return: Modified observation
        """
        obs = self.env.reset(**kwargs)
        obs = self._transform_observation(obs)

        return obs

    def step(self, action):
        """
        Wrapper for the step function
        :param action: action to be taken
        :return: modified obs, reward, done, info
        """
        obs, reward, done, info = self.env.step(action)
        obs = self._transform_observation(obs)

        # maybe TODO: had a note from previous: should I do reward clipping?

        return obs, reward, done, info

    def render(self, tile_size=8):
        """
        Provide rbg rendering of the environment
        :param tile_size: number of pixel to use to render a single tile
        :return: np.ndarray of the current rgb image frame, of shape
                 (channel, height, width), or
                 (3, tiles*tile_size, tiles*tile_size)
        """
        # Render RGB image
        rgb_img = self.env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=tile_size
        )  # (height, width, channel)

        rgb_img = np.moveaxis(rgb_img, 2, 0)  # (channel, height, width)

        return rgb_img

    # potnetial TODO: add something to go from the observation space to image space for
    # printing image?
