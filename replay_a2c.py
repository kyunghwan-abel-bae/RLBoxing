import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from a2cagent import A2CAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation, SkipFrame

from gym import spaces

from utils import *

# env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
env = gym.make('BoxingDeterministic-v4', render_mode="human")
# env = gym.make('BoxingNoFrameskip-v4', render_mode="rgb_array")

num_frames = 4

# env = CustomActionSpaceWrapper(env)

env = AdapterGrayScaleObservation(env)
# env = SkipFrame(env, skip=2)
print(f"1 env state : {env.observation_space}")
env = GrayScaleObservation(env, keep_dim=False)
print(f"2 env state : {env.observation_space}")
env = ResizeObservation(env, shape=84)
print(f"3 env state : {env.observation_space}")
env = TransformObservation(env, f=lambda x: x / 255.)
print(f"4 env state : {env.observation_space}")
env = FrameStack(env, num_stack=num_frames)
print(f"5 env state : {env.observation_space}")

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
checkpoint = Path('checkpoints/target/boxing_a2c_net2.ckpt')

# 16 : batch size
agent = A2CAgent(state_dim=(num_frames, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint, func_print=bprint)

logger = MetricLogger(save_dir)

episodes_start = 0
# if checkpoint:
#     episodes_start = agent.data_load.get("episode") + 1

episodes = 5000
best_score = 0
best_e = 0
for e in range(episodes_start, episodes):
    state = env.reset()
    total_reward = 0

    actor_losses, critic_losses, scores = [], [], []

    while True:
        env.render()

        action = agent.act([state], False)

        next_state, reward, done, info = env.step(action)

        total_reward += reward if reward > 0 else 0

        # agent.update_replay_memory(state, action, reward, next_state, done)

        # actor_loss, critic_loss = agent.learn()#state, action, reward, next_state, done)

        # actor_losses.append(actor_loss)
        # critic_losses.append(critic_loss)

        state = next_state

        if done or (total_reward > 99):
            break

    if best_score < total_reward:
        best_score = total_reward
        best_e = e

    # mean_actor_losses = np.mean(actor_losses)
    # mean_critic_losses = np.mean(critic_losses)
    # bprint(f"[episode {e}] best_score at {best_e} : {best_score}, total_reward : {total_reward}, actor_losses : {mean_actor_losses}, critic_losses : {mean_critic_losses}, lr : {agent.optimizer.param_groups[0]['lr']}")

    # agent.scheduler.step()

    if total_reward > 99:
        bprint("KNOCK OUT")
        # agent.write_summary(total_reward, mean_actor_losses, mean_critic_losses, e)

        # agent.save_model(e)
        break

    # if e % 3 == 0:
    # if e % 50 == 0:
    #     print(f"total reward : {total_reward}")#, str_temp)

        # capture_state(state, e)

        # agent.write_summary(total_reward, actor_loss, critic_loss, e)
        # agent.save_model(e)

        # date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # filename = f"A2C_Log_[init_lr_{agent.init_lr}][min_lr_{agent.min_lr}].txt"
        # save_bprint(filename)
