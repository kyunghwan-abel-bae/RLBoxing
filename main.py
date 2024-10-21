import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

# from a2cagent import A2CAgent
from enagent import EnAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation, SkipFrame

from gym import spaces

from utils import *

from collections import deque


class CustomActionSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(CustomActionSpaceWrapper, self).__init__(env)

        # When the observation issues are occurred, then customize below.
        # self.observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.int32)

        # Define the new action space, for example restricting the actions to 0, 1, and 2
        # self.action_space = spaces.Discrete(8)

    def action(self, act):
        original_action = act + 10
        return original_action


def capture_state(input, ep):
    count = input.shape[0]
    fig, ax = plt.subplots(1, count, figsize=(count * 2, 2))
    axes = ax.flatten()

    for i in range(count):
        axes[i].imshow(input[i])
        axes[i].axis('off')

    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"{date_time}_{ep}_{count}stacks"

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()


env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
# env = gym.make('BoxingDeterministic-v4', render_mode="human")
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

checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

# 16 : batch size
agent = EnAgent(state_dim=(num_frames, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint, func_print=bprint)

logger = MetricLogger(save_dir)

episodes_start = 0
if checkpoint:
    episodes_start = agent.data_load.get("episode") + 1

episodes = 30
best_score = 0
best_e = 0

last_3_total_rewards = deque(maxlen=4)
knock_out_count = 0

interval_init = 3#5
interval_target = 9#20
enable_target_annealing = False

for e in range(episodes_start, episodes):
    state = env.reset()
    total_reward = 0

    actor_losses, critic_losses, scores = [], [], []

    if e % interval_init == 0:
        enable_target_annealing = False
        agent.init_model_weights()

    if e % interval_target == 0:
        enable_target_annealing = True
        agent.update_target()

    while True:
        action = agent.act([state], enable_target_annealing)

        next_state, reward, done, info = env.step(action)

        total_reward += reward if reward > 0 else 0

        # print(f"action : {action}")

        agent.update_replay_memory(state, action, reward, next_state, done)

        actor_loss, critic_loss = agent.learn(enable_target_annealing)#state, action, reward, next_state, done)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        state = next_state

        if done or (total_reward > 99):
            break

    if best_score < total_reward:
        best_score = total_reward
        best_e = e

    mean_actor_losses = np.mean(actor_losses)
    mean_critic_losses = np.mean(critic_losses)
    bprint(f"[episode {e}] best_score at {best_e} : {best_score}, knockout_count : {knock_out_count}, total_reward : {total_reward}, actor_losses : {mean_actor_losses}, critic_losses : {mean_critic_losses}, lr : {agent.optimizer.param_groups[0]['lr']}")
    last_3_total_rewards.append(total_reward)

    # agent.scheduler.step()

    if total_reward > 99:
        bprint("KNOCK OUT")
        knock_out_count += 1

        for param_group in agent.optimizer.param_groups:
            param_group['lr'] *= 0.5  # 새로운 학습률

        agent.write_summary(total_reward, mean_actor_losses, mean_critic_losses, e)

        agent.save_model(e)

        bprint(f"sum(last 3 total rewards) : {sum(last_3_total_rewards)}")
        if sum(last_3_total_rewards) > 340:
            break

    # if e % 3 == 0:
    if e % 50 == 0:
        print(f"total reward : {total_reward}")#, str_temp)

        capture_state(state, e)

        agent.write_summary(total_reward, actor_loss, critic_loss, e)
        agent.save_model(e)

        # date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"[Replay&Conv]A2C_Log_[{num_frames}stack][init_lr_{agent.init_lr}][min_lr_{agent.min_lr}].txt"
        save_bprint(filename)
