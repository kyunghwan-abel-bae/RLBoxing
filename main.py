import argparse
import datetime
import torch
from pathlib import Path
import itertools

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


def capture_state(input_data, name):
    # Ensure the input is a numpy array
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.cpu().numpy()

    # Handle frame stacks
    if len(input_data.shape) == 3:
        count = input_data.shape[0]
        fig, axes = plt.subplots(1, count, figsize=(count * 2, 2))
        if count == 1:
            axes = [axes]
        for i in range(count):
            axes[i].imshow(input_data[i], cmap='gray')
            axes[i].axis('off')
    # Handle single frames
    elif len(input_data.shape) == 2:
        count = 1
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.imshow(input_data, cmap='gray')
        ax.axis('off')
    else:
        print(f"capture_state: Invalid input shape {input_data.shape}")
        return

    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    filename = screenshots_dir / f"{name}_{count}stacks.png"

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run RL experiments with different hyperparameters.")
    parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cpu', 'cuda', 'mps'], help="Device to use for training.")
    args = parser.parse_args()

    # --- Hyperparameter Setup ---
    list_stacks = [2, 3]
    list_steps = [1, 2, 3, 4, 5]
    episodes_per_experiment = 300

    # --- Device Setup ---
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"{ '='*20}\n  Using device: {device}\n{'='*20}")
    # --------------------

    # --- Main Experiment Loop ---
    for num_frames, n_step in itertools.product(list_stacks, list_steps):
        print(f"\n\n{'='*50}")
        print(f"  Starting Experiment: FrameStack={num_frames}, n-step={n_step}")
        print(f"{ '='*50}\n")

        # Set seed for reproducibility for each experiment
        SEED = 0
        seed_all(SEED)

        # --- Environment Setup ---
        env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
        env = AdapterGrayScaleObservation(env)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = TransformObservation(env, f=lambda x: x / 255.)
        env = FrameStack(env, num_stack=num_frames)
        env.reset()

        # --- Directory Setup for this experiment ---
        experiment_name = f"stack_{num_frames}_nstep_{n_step}"
        save_dir = Path("checkpoints") / experiment_name / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        save_dir.mkdir(parents=True)

        # --- Agent Initialization ---
        agent = SACAgent(
            state_dim=(num_frames, 84, 84),
            action_dim=env.action_space.n,
            save_dir=save_dir,
            device=device,
            n_step=n_step,
            actor_lr=3e-5,
            critic_lr=3e-5,
            alpha_tuning_start_episode=100,
            capture_state_func=capture_state,
            capture_episode_freq=50
        )

        logger = MetricLogger(save_dir)

        # --- Training Loop for this experiment ---
        episodes_start = 0
        best_score = 0
        best_e = 0
        last_3_total_rewards = deque(maxlen=4)
        knock_out_count = 0

        for e in range(episodes_start, episodes_per_experiment):
            state, info = env.reset()
            total_reward = 0
            step_count = 0
            actor_losses, critic_losses, scores = [], [], []

            while True:
                step_count += 1
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward if reward > 0 else 0
                agent.update_replay_memory(state, action, reward, next_state, done, e, step_count)
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
            bprint(f"[Exp: {experiment_name}] [episode {e}] best_score at {best_e} : {best_score}, knockout_count : {knock_out_count}, total_reward : {total_reward}, "
                   f"actor_loss : {mean_actor_loss:.4f}, critic_loss : {mean_critic_loss:.4f}, "
                   f"alpha: {alpha_value:.4f}")
            last_3_total_rewards.append(total_reward)

            if total_reward > 99:
                bprint("KNOCK OUT")
                knock_out_count += 1
                agent.save_model(e)
                bprint(f"sum(last 3 total rewards) : {sum(last_3_total_rewards)}")

            if e % 50 == 0 and e > 0:
                print(f"total reward : {total_reward}")
                agent.save_model(e)
        
        env.close()

if __name__ == '__main__':
    main()