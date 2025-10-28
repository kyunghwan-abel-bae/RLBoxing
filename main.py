import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation

from sacagent import SACAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation, SkipFrame

from gym import spaces

from utils import *

# Set seed for reproducibility
SEED = 0
seed_all(SEED)

from collections import deque

import matplotlib.pyplot as plt

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
agent = SACAgent(state_dim=(num_frames, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, actor_lr=3e-6, critic_lr=3e-5, alpha_tuning_start_episode=500)

logger = MetricLogger(save_dir)

episodes_start = 0
if checkpoint:
    episodes_start = agent.data_load.get("episode") + 1

episodes = 3000
best_score = 0
best_e = 0

last_3_total_rewards = deque(maxlen=4)
knock_out_count = 0

for e in range(episodes_start, episodes):
    state, info = env.reset()
    total_reward = 0

    actor_losses, critic_losses, scores = [], [], []

    while True:
        # 1. Select action
        action = agent.act(state)

        # 2. Act in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward if reward > 0 else 0

        # 3. Save experience
        agent.update_replay_memory(state, action, reward, next_state, done)

        # 4. Learn from experiences
        learn_result = agent.learn(e)

        # Only record losses if learning has started
        if learn_result and learn_result[0] is not None:
            actor_loss, critic_loss = learn_result
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        # 5. Update state
        state = next_state

        if done or (total_reward > 99):
            break

    if best_score < total_reward:
        best_score = total_reward
        best_e = e

    # Calculate mean losses only if there were learning steps
    mean_actor_loss = np.mean(actor_losses) if actor_losses else 0
    mean_critic_loss = np.mean(critic_losses) if critic_losses else 0

    # Log metrics
    if mean_actor_loss != 0 and mean_critic_loss != 0:
        agent.write_summary(total_reward, mean_actor_loss, mean_critic_loss, agent.alpha.item(), e)

    bprint(f"[episode {e}] best_score at {best_e} : {best_score}, knockout_count : {knock_out_count}, total_reward : {total_reward}, "
           f"actor_loss : {mean_actor_loss:.4f}, critic_loss : {mean_critic_loss:.4f}, "
           f"alpha: {agent.alpha.item():.4f}")
    last_3_total_rewards.append(total_reward)

    if total_reward > 99:
        bprint("KNOCK OUT")
        knock_out_count += 1
        # The A2C-specific learning rate decay is removed as it's not standard for SAC with Adam.

        agent.save_model(e) # Save model on knockout

        bprint(f"sum(last 3 total rewards) : {sum(last_3_total_rewards)}")
        if sum(last_3_total_rewards) > 340:
            break

    if e % 50 == 0 and e > 0:
        print(f"total reward : {total_reward}")

        capture_state(state, e)

        agent.save_model(e) # Save model periodically

        # The filename logic was specific to A2C and has been simplified.
        # filename = f"[SAC]_Log_[{num_frames}stack].txt"
        # save_bprint(filename)
