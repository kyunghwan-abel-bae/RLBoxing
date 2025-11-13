import argparse
import torch
from pathlib import Path
import time

import gymnasium as gym
from gymnasium.wrappers import FrameStack, TransformObservation, GrayScaleObservation, RecordVideo

from sacagent import SACAgent
from wrappers import ResizeObservation, AdapterGrayScaleObservation
from utils import seed_all

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Replay a trained SAC agent.")
    parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cpu', 'cuda', 'mps'], help="Device to use for replay.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/target/sac_agent.ckpt", help="Path to the agent checkpoint file.")
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1], help="Set in-game difficulty (0=B/easy, 1=A/hard).")
    parser.add_argument("--record", type=str, default=None, help="Directory to save video replay (e.g., 'videos/').")
    args = parser.parse_args()

    # --- Device Setup ---
    device = args.device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"{ '='*20}\n  Using device: {device}\n{ '='*20}")
    # --------------------

    # --- Hyperparameters from training ---
    # These must match the conditions under which the agent was trained.
    num_frames = 2
    n_step = 1
    actor_lr = 3e-05
    critic_lr = 3e-05

    # Set seed for reproducibility
    seed_all(0)

    # --- Environment Setup ---
    # --- Environment Setup ---
    render_mode = "rgb_array" if args.record else "human"
    env = gym.make(
        'BoxingDeterministic-v4',
        render_mode=render_mode,
        difficulty=args.difficulty
    )

    try:
        # Apply video recording wrapper if specified
        if args.record:
            video_path = Path(args.record)
            video_path.mkdir(parents=True, exist_ok=True)
            # Record every episode
            env = RecordVideo(env, video_folder=str(video_path), episode_trigger=lambda e: True)
            print(f"... Recording video to {video_path.resolve()}. Live rendering will be disabled.")

        env = AdapterGrayScaleObservation(env)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = TransformObservation(env, f=lambda x: x / 255.)
        env = FrameStack(env, num_stack=num_frames)
        env.reset()

        # --- Agent Initialization ---
        agent = SACAgent(
            state_dim=(num_frames, 84, 84),
            action_dim=env.action_space.n,
            save_dir=Path("replay_logs"), # A dummy save_dir for the agent
            device=device,
            n_step=n_step,
            actor_lr=actor_lr,
            critic_lr=critic_lr
        )

        # --- Load Checkpoint ---
        try:
            agent.load_model(args.checkpoint)
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {args.checkpoint}")
            return

        # --- Replay Loop ---
        for e in range(10): # Replay for 10 episodes
            state, info = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Use training=False to ensure deterministic actions (if applicable) and no learning
                action = agent.act(state, training=False)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

                # Optional: sleep to make the replay slower and more watchable
                time.sleep(0.01)

            print(f"Episode {e+1}: Total Reward: {total_reward}")
    finally:
        env.close()

if __name__ == '__main__':
    main()