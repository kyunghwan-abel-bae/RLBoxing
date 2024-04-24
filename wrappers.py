import gym
import torch
import random, datetime, numpy as np
from skimage import transform

from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        obs, reward, done, truncated, info = self.env.step(action)

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs[0]