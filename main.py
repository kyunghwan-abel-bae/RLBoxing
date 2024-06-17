import os

import datetime
from pathlib import Path

import gymnasium as gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from a2cagent import A2CAgent
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

checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

agent = A2CAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint)
actor_losses, critic_losses, scores = [], [], []

logger = MetricLogger(save_dir)

episodes = 5000
for e in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        print(f"state : {state.shape}")
        action = agent.act(state)
        print(f"action : {action}")
        quit()

        next_state, reward, done, info = env.step(action)

        total_reward += reward if reward > 0 else 0

        q, loss = agent.learn()

        logger.log_step(total_reward, loss, q)

        state = next_state

        if done or (total_reward > 99):
            break

    logger.log_episode()

    if total_reward > 99:
        print("KNOCK OUT")
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

        agent.save()
        break

    if e % 50 == 0:
        print(f"total reward : {total_reward}")
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

        agent.save()

