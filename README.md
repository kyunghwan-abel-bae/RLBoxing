# RLBoxing: A Deep Reinforcement Learning Project

This project implements a deep reinforcement learning agent to play the game of boxing. The agent is trained using the Soft Actor-Critic (SAC) algorithm. This README file provides a log of the development process, including key architectural changes, hyperparameter tuning, and feature additions.

## Key Development Stages

The development of the RLBoxing agent can be broadly categorized into the following stages:

1.  **Initial A2C Implementation:** The project started with an implementation of the Advantage Actor-Critic (A2C) algorithm. This involved setting up the basic environment, agent, and training loop.
2.  **Migration to SAC:** To improve performance and sample efficiency, the agent was migrated from A2C to Soft Actor-Critic (SAC). This was a major architectural change that required rewriting the agent's core logic, including the network architecture and learning algorithm.
3.  **Hyperparameter Tuning and Stabilization:** A significant amount of time was spent on tuning hyperparameters to stabilize the training process and improve the agent's performance. This included experimenting with different learning rates, target entropy, and the temperature parameter alpha.
4.  **Advanced Techniques:** Several advanced reinforcement learning techniques were implemented to further improve the agent's performance, such as N-step returns and gradient clipping.
5.  **Feature Additions:** Additional features were added to improve the project's usability and experimentation capabilities, such as automatic hyperparameter tuning, different difficulty modes, and a function to record gameplay videos.

---

## Detailed Commit Log

This section provides a detailed log of the project's commit history, with explanations for the most significant changes.

-   `b5e001d` - **gitignore added (2025-11-15):** Added a `.gitignore` file to exclude unnecessary files from the repository.
-   `c789fa5` - **record function added (2025-11-13):** Implemented a feature to record gameplay videos using the `--record` flag. This is useful for visualizing the agent's performance and debugging.
-   `34ccc34` - **difficulty mode hyperparameters added (2025-11-10):** Introduced different sets of hyperparameters for different difficulty modes. This allows for more targeted training and evaluation.
-   `db3911e` - **replay_sac added (2025-11-09):** Added a script to replay a trained SAC agent.
-   `9207a8e` - **specify data type as float32 (2025-11-08):** Explicitly set the data type of tensors to `float32` for consistency and to prevent potential errors.
-   `294139a` - **auto hyperparameter tuning ready & break time added (2025-11-07):** Implemented a framework for automatic hyperparameter tuning. This allows for systematically searching for the optimal set of hyperparameters. A break time was also added to the training loop to prevent overheating.
-   `5d8c2b5` - **gradient clipping added (2025-11-02):** Implemented gradient clipping to prevent exploding gradients and stabilize the training process.
-   `78cf173` - **lr & alpha refined (2025-10-30):** Refined the learning rate and the temperature parameter alpha based on experimental results.
-   `c5d8227` - **target entropy refined (2025-10-30):** Adjusted the target entropy value to better balance exploration and exploitation.
-   `6aee8ea` - **n step return adapted (2025-10-27):** Implemented N-step returns to improve the agent's learning speed and performance. This technique allows the agent to learn from rewards that are multiple steps in the future.
-   `3c51246` & `39be1df` - **two lr (2025-10-26):** Separated the learning rates for the actor and critic networks. This allows for more fine-grained control over the learning process and can lead to better performance.
-   `863d454` - **running without errors (2025-10-23):** A stable version of the code that runs without errors.
-   `bc52824` - **DoubleQNetwork is adapted (2025-10-22):** Implemented a Double Q-Network to reduce the overestimation bias of the critic.
-   `397299d` - **api updated for gymnasium, replay_a2c.py runs well (2025-10-15):** Migrated the project from `gym` to `gymnasium` to resolve API compatibility issues. This was a necessary step to ensure the project's long-term maintainability.
-   `6af38bc` - **[SAC Implemeted with Windsurf] (2024-11-26):** This was the initial implementation of the Soft Actor-Critic (SAC) algorithm. The network architecture was changed from a shared network to individual networks for the actor and critic.
-   `4398224` - **network optimized, double channels in conv (2024-08-30):** Optimized the network architecture by doubling the number of channels in the convolutional layers.
-   `368f774` - **replay for a2c added (2024-08-26):** Implemented a replay buffer for the A2C agent.
-   `cccf46f` - **init (2024-04-23):** Initial commit.

---
# RLBoxing: 심층 강화 학습 프로젝트

이 프로젝트는 복싱 게임을 플레이하는 심층 강화 학습 에이전트를 구현합니다. 에이전트는 SAC(Soft Actor-Critic) 알고리즘을 사용하여 훈련됩니다. 이 README 파일은 주요 아키텍처 변경, 하이퍼파라미터 튜닝 및 기능 추가를 포함한 개발 프로세스 로그를 제공합니다.

## 주요 개발 단계

RLBoxing 에이전트의 개발은 크게 다음과 같은 단계로 나눌 수 있습니다.

1.  **초기 A2C 구현:** 프로젝트는 A2C(Advantage Actor-Critic) 알고리즘 구현으로 시작되었습니다. 여기에는 기본 환경, 에이전트 및 훈련 루프 설정이 포함되었습니다.
2.  **SAC로 마이그레이션:** 성능과 샘플 효율성을 개선하기 위해 에이전트를 A2C에서 SAC(Soft Actor-Critic)로 마이그레이션했습니다. 이는 네트워크 아키텍처 및 학습 알고리즘을 포함한 에이전트의 핵심 로직을 다시 작성해야 하는 주요 아키텍처 변경이었습니다.
3.  **하이퍼파라미터 튜닝 및 안정화:** 훈련 과정을 안정화하고 에이전트의 성능을 향상시키기 위해 하이퍼파라미터를 튜닝하는 데 상당한 시간을 할애했습니다. 여기에는 다양한 학습률, 목표 엔트로피 및 온도 매개변수 알파를 실험하는 것이 포함되었습니다.
4.  **고급 기술:** N-step 리턴 및 그래디언트 클리핑과 같은 여러 고급 강화 학습 기술을 구현하여 에이전트의 성능을 더욱 향상시켰습니다.
5.  **기능 추가:** 자동 하이퍼파라미터 튜닝, 다양한 난이도 모드 및 게임 플레이 비디오 녹화 기능과 같은 추가 기능이 추가되어 프로젝트의 사용성과 실험 기능을 개선했습니다.

---

## 상세 커밋 로그

이 섹션에서는 가장 중요한 변경 사항에 대한 설명과 함께 프로젝트의 커밋 기록에 대한 자세한 로그를 제공합니다.

-   `b5e001d` - **gitignore 추가 (2025-11-15):** 저장소에서 불필요한 파일을 제외하기 위해 `.gitignore` 파일을 추가했습니다.
-   `c789fa5` - **녹화 기능 추가 (2025-11-13):** `--record` 플래그를 사용하여 게임 플레이 비디오를 녹화하는 기능을 구현했습니다. 이는 에이전트의 성능을 시각화하고 디버깅하는 데 유용합니다.
-   `34ccc34` - **난이도 모드 하이퍼파라미터 추가 (2025-11-10):** 다양한 난이도 모드에 대해 다른 하이퍼파라미터 세트를 도입했습니다. 이를 통해 보다 목표에 맞는 훈련 및 평가가 가능합니다.
-   `db3911e` - **replay_sac 추가 (2025-11-09):** 훈련된 SAC 에이전트를 재생하는 스크립트를 추가했습니다.
-   `9207a8e` - **데이터 유형을 float32로 지정 (2025-11-08):** 일관성을 유지하고 잠재적인 오류를 방지하기 위해 텐서의 데이터 유형을 `float32`로 명시적으로 설정했습니다.
-   `294139a` - **자동 하이퍼파라미터 튜닝 준비 및 휴식 시간 추가 (2025-11-07):** 자동 하이퍼파라미터 튜닝을 위한 프레임워크를 구현했습니다. 이를 통해 최적의 하이퍼파라미터 세트를 체계적으로 검색할 수 있습니다. 과열을 방지하기 위해 훈련 루프에 휴식 시간도 추가되었습니다.
-   `5d8c2b5` - **그래디언트 클리핑 추가 (2025-11-02):** 그래디언트 폭발을 방지하고 훈련 과정을 안정화하기 위해 그래디언트 클리핑을 구현했습니다.
-   `78cf173` - **lr 및 alpha 미세 조정 (2025-10-30):** 실험 결과를 바탕으로 학습률과 온도 매개변수 알파를 미세 조정했습니다.
-   `c5d8227` - **목표 엔트로피 미세 조정 (2025-10-30):** 탐색과 활용의 균형을 더 잘 맞추기 위해 목표 엔트로피 값을 조정했습니다.
-   `6aee8ea` - **n-step 리턴 적용 (2025-10-27):** 에이전트의 학습 속도와 성능을 향상시키기 위해 N-step 리턴을 구현했습니다. 이 기술을 통해 에이전트는 미래의 여러 단계에 있는 보상으로부터 학습할 수 있습니다.
-   `3c51246` & `39be1df` - **두 개의 lr (2025-10-26):** 액터 및 크리틱 네트워크의 학습률을 분리했습니다. 이를 통해 학습 과정을 보다 세밀하게 제어할 수 있으며 더 나은 성능을 얻을 수 있습니다.
-   `863d454` - **오류 없이 실행 (2025-10-23):** 오류 없이 실행되는 안정적인 버전의 코드입니다.
-   `bc52824` - **DoubleQNetwork 적용 (2025-10-22):** 크리틱의 과대평가 편향을 줄이기 위해 Double Q-Network를 구현했습니다.
-   `397299d` - **gymnasium용 api 업데이트, replay_a2c.py 정상 실행 (2025-10-15):** API 호환성 문제를 해결하기 위해 프로젝트를 `gym`에서 `gymnasium`으로 마이그레이션했습니다. 이는 프로젝트의 장기적인 유지 관리를 보장하기 위해 필요한 단계였습니다.
-   `6af38bc` - **[Windsurf로 SAC 구현] (2024-11-26):** 이것은 SAC(Soft Actor-Critic) 알고리즘의 초기 구현이었습니다. 네트워크 아키텍처는 공유 네트워크에서 액터와 크리틱을 위한 개별 네트워크로 변경되었습니다.
-   `4398224` - **네트워크 최적화, conv의 채널 두 배로 (2024-08-30):** 컨볼루션 레이어의 채널 수를 두 배로 늘려 네트워크 아키텍처를 최적화했습니다.
-   `368f774` - **a2c용 리플레이 추가 (2024-08-26):** A2C 에이전트를 위한 리플레이 버퍼를 구현했습니다.
-   `cccf46f` - **초기화 (2024-04-23):** 초기 커밋.