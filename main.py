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

num_frames = 3

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
checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

agent = A2CAgent(state_dim=(num_frames, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes_start = 0
if checkpoint:
    episodes_start = agent.data_load.get("episode") + 1

episodes = 5000
best_score = 0
for e in range(episodes_start, episodes):
    state = env.reset()
    total_reward = 0

    actor_losses, critic_losses, scores = [], [], []

    while True:
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        total_reward += reward if reward > 0 else 0

        actor_loss, critic_loss = agent.learn(state, action, reward, next_state, done)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        state = next_state

        if done or (total_reward > 99):
            break

    best_score = total_reward if best_score < total_reward else best_score
    mean_actor_losses = np.mean(actor_losses)
    mean_critic_losses = np.mean(critic_losses)
    print(f"[episode {e}] best_score : {best_score}, total_reward : {total_reward}, actor_losses : {mean_actor_losses}, critic_losses : {mean_critic_losses}, lr : {agent.optimizer.param_groups[0]['lr']}")

    agent.scheduler.step()


    if total_reward > 99:
        print("KNOCK OUT")
        agent.write_summary(total_reward, mean_actor_losses, mean_critic_losses, e)

        agent.save_model(e)
        break

    if e % 50 == 0:
        print(f"total reward : {total_reward}")

        capture_state(state, e)

        agent.write_summary(total_reward, actor_loss, critic_loss, e)
        agent.save_model(e)


