import datetime
import torch
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation

from sacagent import SACAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation, SkipFrame

from gym import spaces

from utils import *

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


def main():
    # --- Device Setup ---
    choice = input("Enter device to use (auto, cpu, cuda, mps) [default: auto]: ").strip().lower()
    if not choice:
        choice = 'auto'

    device = choice
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"{'='*20}\n  Using device: {device}\n{'='*20}")
    # --------------------

    # Set seed for reproducibility
    SEED = 0
    seed_all(SEED)

    env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")

    num_frames = 4

    env = AdapterGrayScaleObservation(env)
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

    checkpoint = None

    agent = SACAgent(
        state_dim=(num_frames, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        device=device,
        n_step=1,
        actor_lr=3e-5,
        critic_lr=3e-5,
        alpha_tuning_start_episode=100
    )

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
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward if reward > 0 else 0
            agent.update_replay_memory(state, action, reward, next_state, done)
            learn_result = agent.learn(e)

            if learn_result and learn_result[0] is not None:
                actor_loss, critic_loss = learn_result
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            state = next_state

            if done or (total_reward > 99):
                break

        if best_score < total_reward:
            best_score = total_reward
            best_e = e

        mean_actor_loss = np.mean(actor_losses) if actor_losses else 0
        mean_critic_loss = np.mean(critic_losses) if critic_losses else 0

        if mean_actor_loss != 0 and mean_critic_loss != 0:
            alpha_value = agent.alpha.item() if torch.is_tensor(agent.alpha) else agent.alpha
            agent.write_summary(total_reward, mean_actor_loss, mean_critic_loss, alpha_value, e)

        alpha_value = agent.alpha.item() if torch.is_tensor(agent.alpha) else agent.alpha
        bprint(f"[episode {e}] best_score at {best_e} : {best_score}, knockout_count : {knock_out_count}, total_reward : {total_reward}, "
               f"actor_loss : {mean_actor_loss:.4f}, critic_loss : {mean_critic_loss:.4f}, "
               f"alpha: {alpha_value:.4f}")
        last_3_total_rewards.append(total_reward)

        if total_reward > 99:
            bprint("KNOCK OUT")
            knock_out_count += 1
            agent.save_model(e)
            bprint(f"sum(last 3 total rewards) : {sum(last_3_total_rewards)}")
            if sum(last_3_total_rewards) > 340:
                break

        if e % 50 == 0 and e > 0:
            print(f"total reward : {total_reward}")
            capture_state(state, e)
            agent.save_model(e)

if __name__ == '__main__':
    main()
