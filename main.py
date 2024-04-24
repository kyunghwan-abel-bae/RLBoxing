import os

import datetime
from pathlib import Path

import gymnasium as gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame

import numpy as np

env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
# env = gym.make('BoxingNoFrameskip-v4', render_mode="rgb_array")

env.observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.int)

env = SkipFrame(env, skip=1)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

# agent

logger = MetricLogger(save_dir)

episodes = 100

for e in range(episodes):
    state = env.reset()
    print(f"state.shape : {state[0].shape}")
    break