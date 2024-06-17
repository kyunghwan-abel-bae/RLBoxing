import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from a2cagent import A2CAgent
from metrics import MetricLogger
from wrappers import ResizeObservation, AdapterGrayScaleObservation

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

    actor_loss, critic_loss = 0

    while True:
        action = agent.act(state)

        # print(f"state : {state}")

        next_state, reward, done, info = env.step(action)
        # next_state = np.array(next_state)
        # print(f"next_state : {next_state}, reward : {reward}, done : {done}")
        # print("point1")
        total_reward += reward if reward > 0 else 0

        q, loss = agent.learn(state, action, reward, next_state, done)
        # print("point2")
        # quit()

        logger.log_step(total_reward, loss, q)

        state = next_state

        # print("point5")

        if done or (total_reward > 99):
            break

    logger.log_episode()

    if total_reward > 99:
        print("KNOCK OUT")

        #     def write_summary(self, score, actor_loss, critic_loss, step):
        agent.write_summary(total_reward)

        agent.save_model()

        break

    if e % 50 == 0:
        print(f"total reward : {total_reward}")
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )

        agent.save()

