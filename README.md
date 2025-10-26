# Log

###Branches
- dev : main branch
- dev-win : [main branch] tested or improved in the win env


2024.04.29
- dev-win is created
- Small replay memory may occur bad learning. This issue can be a key to solve the problem.

---

## 2025-10-23: Debugging & Hyperparameter Tuning Log

SAC 에이전트의 훈련 안정화를 위해 일련의 디버깅 및 하이퍼파라미터 튜닝을 진행했습니다.

### 1. 초기 오류 해결

- **문제:** 라이브러리 비호환성으로 인해 스크립트 실행 실패.
- **해결:**
    - `gym`에서 `gymnasium`으로 마이그레이션하여 래퍼(wrapper), `env.reset()`, `env.step()`의 API 충돌 문제 해결.
    - `TransformObservation`, `numpy` import 누락으로 인한 `NameError` 해결.
    - `agent.learn()` 함수가 `(None, None)`을 반환할 때 이를 처리하는 로직을 추가하여 `TypeError` 해결.

### 2. 안정성을 위한 하이퍼파라미터 튜닝

- **관찰 1: Critic Loss 폭발**
    - **시도:** 학습률(learning rate)을 `3e-6`까지 크게 낮춤.
    - **결과:** Loss는 안정화되었으나, 에이전트의 탐험(exploration) 능력이 크게 저하됨.

- **관찰 2: 부족한 탐험**
    - **시도:** 초기 탐험을 장려하기 위해 `sacagent.py`에서 초기 온도 파라미터 `alpha` 값을 `0.01`에서 `0.2`로 상향 조정.
    - **결과:** 초기 탐험은 증가했으나, 약 500 에피소드 이후 훈련이 불안정해짐.

- **관찰 3: 불안정한 수렴**
    - **증상:** Critic loss와 `alpha` 값이 최소점을 찍은 후 다시 상승하는 현상 발생.
    - **분석:** Actor와 Critic이 동일한 학습률로 업데이트되면서 불안정한 피드백 루프에 빠진 것으로 추정. Critic이 빠르게 변하는 정책에 대한 안정적인 가치 함수를 학습하지 못함.
    - **해결:** Actor와 Critic의 학습률을 분리. Critic이 Actor보다 더 빠르게 학습하도록 설정하여 가치 함수를 안정적으로 수렴시키는 것을 목표로 함. `main.py`에서 아래와 같이 최종 설정 적용.
        - **Actor 학습률 (`actor_lr`):** `3e-6`
        - **Critic 학습률 (`critic_lr`):** `3e-5`
