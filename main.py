import datetime
from pathlib import Path

import gymnasium as gym
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

        # Define the new action space, for example restricting the actions to 0, 1, and 2
        self.action_space = spaces.Discrete(8)

        # self.env.observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.int32)

    def action(self, action):
        # Map the custom action space to the original action space
        # For example, we might map actions {0, 1, 2} to original actions {0, 1, 2}
        # You can customize this mapping based on your needs
        original_action = action + 10
        return original_action


env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")
# env = gym.make('BoxingDeterministic-v4', render_mode="human")
# env = gym.make('BoxingNoFrameskip-v4', render_mode="rgb_array")

env = CustomActionSpaceWrapper(env)

# noted by KH -- Below need to be moved to the above class
env.observation_space = Box(low=0, high=255, shape=(210, 160, 3), dtype=np.int32)

# env = AdapterGrayScaleObservation(env)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')
checkpoint = None  # Path('checkpoints/2020-10-21T18-25-27/mario.chkpt')

agent = A2CAgent(state_dim=(4, 84, 84), action_dim=env.action_space.n, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes_start = 0
if checkpoint:
    episodes_start = agent.data_load.get("episode") + 1

episodes = 5000
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

    mean_actor_losses = np.mean(actor_losses)
    mean_critic_losses = np.mean(critic_losses)
    print(f"[episode {e}] total_reward : {total_reward}, actor_losses : {mean_actor_losses}, critic_losses : {mean_critic_losses}, lr : {agent.optimizer.param_groups[0]['lr']}")

    agent.scheduler.step()

    if total_reward > 99:
        print("KNOCK OUT")
        agent.write_summary(total_reward, mean_actor_losses, mean_critic_losses, e)

        agent.save_model(e)
        break

    if e % 50 == 0:
        print(f"total reward : {total_reward}")

        agent.write_summary(total_reward, actor_loss, critic_loss, e)
        agent.save_model(e)


