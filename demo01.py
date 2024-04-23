import gymnasium as gym
import pygame
import numpy as np

render = True

env = gym.make('BoxingDeterministic-v4', render_mode="human")
# env = gym.make('BoxingDeterministic-v4', render_mode="rgb_array")

# 게임 루프
running = True

while running:
    # CartPole 상태 업데이트
    observation = env.reset()
    observation = observation[0]

    '''
    print(f"len(observation) : {len(observation)}")
    print(f"observation[0].shape : {observation[0].shape}")
    print(f"observation[1] : {observation[1]}")
    # print(f"observation[0].shape : {observation[0].shape}")

    print(f"env.step(0) : {env.step(0)}")
    print(f"len(env.step(0)) : {len(env.step(0))}")
    print(f"env.step(0)[0].shape : {env.step(0)[0].shape}")
    '''

    score = 0
    for t in range(1000):  # 최대 1000번 반복
        env.render() # in LOOP

        action = env.action_space.sample()  # 무작위로 행동 선택

        observation, reward, terminated, truncated, info = env.step(action)
        # action test
        # observation, reward, terminated, truncated, info = env.step(1)

        if terminated:
            print("DONE")
            pygame.display.set_caption("DONE")
            running = False
            break

