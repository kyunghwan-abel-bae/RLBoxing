import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import datetime
import random
from collections import deque
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim[1:]
        a = output_dim[1]

        if h != 84 or w != 84:
            raise ValueError(f"Expecting input shape (84, 84), got ({h}, {w})")

        self.cnn_base = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn_base(dummy_input).shape
            flattened_dim = cnn_out_dim[1] * cnn_out_dim[2] * cnn_out_dim[3]

        self.head = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flattened_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, a)
        )

    def forward(self, x):
        x = self.cnn_base(x)
        x = self.head(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim[1:]
        a = output_dim[1]

        if h != 84 or w != 84:
            raise ValueError(f"Expecting input shape (84, 84), got ({h}, {w})")

        self.cnn_base = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.cnn_base(dummy_input).shape
            flattened_dim = cnn_out_dim[1] * cnn_out_dim[2] * cnn_out_dim[3]

        self.head = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flattened_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, a)
        )

    def forward(self, x):
        x = self.cnn_base(x)
        return self.head(x)


class DoubleQNetwork(nn.Module):
    """
    A container for two Q-networks (Critic), as per the user's design.
    Each critic is an independent network with its own CNN encoder.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.q1 = CriticNetwork(input_dim, output_dim)
        self.q2 = CriticNetwork(input_dim, output_dim)
        
    def forward(self, state):
        # Return the Q-values from both critics
        return self.q1(state), self.q2(state)


from torch.utils.tensorboard import SummaryWriter

class SACAgent:
    def __init__(self, state_dim, action_dim, save_dir, device, actor_lr=3e-5, critic_lr=3e-4, gamma=0.99, tau=0.005, n_step=3, fixed_initial_alpha=0.2, alpha_tuning_start_episode=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.save_dir = save_dir
        self.device = device
        self.n_step = n_step
        self.alpha_tuning_start_episode = alpha_tuning_start_episode

        # Replay Memory
        self.min_replay_memory_size = 1000
        self.replay_memory = deque(maxlen=100000)
        self.n_step_buffer = [] # For N-step returns
        self.batch_size = 32
        
        # Network dimensions
        model_state_dim = (self.batch_size,) + self.state_dim
        model_action_dim = (self.batch_size, self.action_dim)

        # Networks
        self.actor = ActorNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.q_network = DoubleQNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.target_q_network = DoubleQNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=critic_lr)

        # Temperature parameter for entropy
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha = torch.tensor(fixed_initial_alpha, dtype=torch.float32, device=self.device) # Start with a fixed alpha tensor
        self.target_entropy = -torch.log(1 / torch.tensor(self.action_dim)) * 0.98
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=critic_lr)

        # For logging
        self.writer = SummaryWriter(self.save_dir)

    def update_replay_memory(self, state, action, reward, next_state, done):
        state_arr = np.array(state, dtype=np.uint8)
        next_state_arr = np.array(next_state, dtype=np.uint8)
        self.n_step_buffer.append((state_arr, action, reward, next_state_arr, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate the n-step return for the oldest transition in the buffer
        n_step_reward = sum([(self.gamma**i) * self.n_step_buffer[i][2] for i in range(self.n_step)])
        
        # The transition to store is the one at the beginning of the buffer
        start_state, action, _, _, _ = self.n_step_buffer[0]
        # The "next_state" for the n-step return is the state after the last transition in the buffer
        _, _, _, end_next_state, end_done = self.n_step_buffer[-1]

        # Add the processed n-step transition to the main replay buffer
        self.replay_memory.append((start_state, action, n_step_reward, end_next_state, end_done))

        # Remove the oldest transition
        self.n_step_buffer.pop(0)

        # If the episode ended, clear the buffer to not carry over transitions between episodes
        if done:
            self.n_step_buffer.clear()

    def _sample_experience(self):
        return random.sample(self.replay_memory, self.batch_size)

    @torch.no_grad()
    def act(self, state, training=True):
        if training and len(self.replay_memory) < self.min_replay_memory_size:
            return random.randrange(self.action_dim)
        
        state_arr = np.array(state, dtype=np.uint8)
        state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        
        self.actor.eval()
        pi = self.actor(state_tensor)
        self.actor.train()

        dist = torch.distributions.Categorical(probs=pi)
        action = dist.sample()
        return action.item()

    def sample_action(self, states):
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs=probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, probs

    def learn(self, current_episode):
        if len(self.replay_memory) < self.batch_size:
            return None, None

        # 1. Sample from replay buffer
        experiences = self._sample_experience()
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 2. Convert to tensors
        states = torch.from_numpy(np.array(states, dtype=np.uint8)).float().to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.uint8)).float().to(self.device) / 255.0
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. Calculate Critic Target (td_target)
        with torch.no_grad():
            _, next_log_probs, next_probs = self.sample_action(next_states)
            q1_target_next, q2_target_next = self.target_q_network(next_states)
            min_q_target_next = torch.min(q1_target_next, q2_target_next)
            
            # Soft state value V(s_{t+n})
            soft_state_value = (next_probs * (min_q_target_next - self.alpha * next_log_probs.unsqueeze(1))).sum(dim=1, keepdim=True)
            
            # The TD target now uses the n-step reward and gamma^n
            td_target = rewards + (1 - dones) * (self.gamma ** self.n_step) * soft_state_value

        # 4. Calculate Critic Loss
        q1_pred, q2_pred = self.q_network(states)
        q1_pred = q1_pred.gather(1, actions)
        q2_pred = q2_pred.gather(1, actions)

        critic_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        # 5. Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 6. Calculate Actor and Alpha Loss
        _, log_probs, probs = self.sample_action(states)
        
        with torch.no_grad():
            q1_all, q2_all = self.q_network(states)
            min_q_all = torch.min(q1_all, q2_all)

        # Actor loss
        actor_loss = (probs * (self.alpha.detach() * log_probs.unsqueeze(1) - min_q_all)).sum(dim=1).mean()
        
        # 7. Update Actor and (conditionally) Alpha
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if current_episode >= self.alpha_tuning_start_episode:
            # Alpha loss
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # 8. Soft update target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        return actor_loss.item(), critic_loss.item()

    def save_model(self, num_episode):
        save_path = self.save_dir / "sac_agent.ckpt"
        print(f"... Save Model to {save_path}")
        torch.save({
            "episode": int(num_episode),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.q_network.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
        }, save_path)

    def write_summary(self, score, actor_loss, critic_loss, alpha, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("loss/actor_loss", actor_loss, step)
        self.writer.add_scalar("loss/critic_loss", critic_loss, step)
        self.writer.add_scalar("param/alpha", alpha, step)
