import os

import datetime
from pathlib import Path

import gymnasium as gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from agent import Agent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation

import numpy as np

env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
# env = gym.make('BoxingDeterministic-v4', render_mode="human")
# env = gym.make('BoxingNoFrameskip-v4', render_mode="rgb_array")

env.observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.int32)

env = AdapterGrayScaleObservation(env)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

checkpoint = None # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = 40000
total_reward = 0
for e in range(episodes):
    state = env.reset()

    while True:
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        total_reward += reward if reward > 0 else 0

        agent.cache(state, next_state, action, reward, done)
        # print(f"reward : {total_reward}")

        q, loss = agent.learn()

        logger.log_step(reward, loss, q)

        state = next_state

        if done or (total_reward > 199):
            break

    logger.log_episode()

    if e % 50 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )