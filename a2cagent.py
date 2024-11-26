import datetime
import random

import torch.cuda
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from einops.layers.torch import Rearrange

from collections import deque


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        b, c, h, w = input_dim
        b, a = output_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(3136, 256, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.pi = nn.Linear(256, a)

    def forward(self, x):
        x = self.cnn(x)
        return F.softmax(self.pi(x), dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        b, c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(3136, 256, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x = self.cnn(x)
        return self.v(x)


class A2CAgent:
    def __init__(self, state_dim, action_dim, checkpoint=None, func_print=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        ## hyper parameters
        self.discount_factor = 0.9
        date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = f"./saved_a2c/{date_time}"
        self.load_path = checkpoint

        ## SAC parameters
        self.target_entropy = -float(self.action_dim)  # -dim(A)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        ## added
        self.min_replay_memory_size = 500
        self.replay_memory = deque(maxlen=10000)
        self.batch_size = 32

        model_state_dim = (self.batch_size,) + self.state_dim
        model_action_dim = (self.batch_size, self.action_dim)
        
        # Separate networks for Actor and Critics
        self.actor = ActorNetwork(model_state_dim, model_action_dim).float()
        self.critic1 = CriticNetwork(model_state_dim).float()
        self.critic2 = CriticNetwork(model_state_dim).float()
        self.critic_target1 = CriticNetwork(model_state_dim).float()
        self.critic_target2 = CriticNetwork(model_state_dim).float()

        # Device selection with priority: CUDA > MPS > CPU
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        
        # Move all networks to device
        self.actor = self.actor.to(self.device)
        self.critic1 = self.critic1.to(self.device)
        self.critic2 = self.critic2.to(self.device)
        self.critic_target1 = self.critic_target1.to(self.device)
        self.critic_target2 = self.critic_target2.to(self.device)
        self.log_alpha = self.log_alpha.to(self.device)

        # Copy critic parameters to targets
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        self.init_lr = 1e-6
        self.min_lr = 1e-7
        self.func_print = func_print

        # Separate optimizers for each network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.init_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.init_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.init_lr)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=1000, eta_min=self.min_lr)

        self.writer = SummaryWriter(self.save_path)
        self.pi_counter = 0

    def update_replay_memory(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).cuda() if torch.cuda.is_available() else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if torch.cuda.is_available() else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if torch.cuda.is_available() else torch.LongTensor([action])
        reward = torch.FloatTensor([reward]).cuda() if torch.cuda.is_available() else torch.FloatTensor([reward])
        done = torch.BoolTensor([done]).cuda() if torch.cuda.is_available() else torch.BoolTensor([done])

        self.replay_memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        self.actor.train(training)
        
        pi = self.actor(torch.FloatTensor(state).to(self.device))

        self.pi_counter += 1
        if self.pi_counter % 500 == 0:
            self.func_print(f"pi : {pi}")
            self.pi_counter = 0

        action = torch.multinomial(pi, num_samples=1).cpu().numpy()[0]
        return action[0]

    def learn(self):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return -1, -1

        samples = random.sample(self.replay_memory, self.batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*samples))

        # Actor forward pass
        pi = self.actor(state)
        log_pi = torch.log(pi + 1e-10)

        # Critic forward passes
        q1 = self.critic1(state)
        q2 = self.critic2(state)
        
        # Target critics
        with torch.no_grad():
            next_pi = self.actor(next_state)
            next_q1 = self.critic_target1(next_state)
            next_q2 = self.critic_target2(next_state)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1-done.float()) * self.discount_factor * next_q

        # Critic losses
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Calculate entropy
        entropy = -(pi * log_pi).sum(dim=-1, keepdim=True)
        
        # Actor loss with entropy regularization
        q_min = torch.min(q1, q2)
        actor_loss = -(q_min * pi).sum(dim=-1).mean() - self.alpha.detach() * entropy.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update of target networks
        self._soft_update_targets()

        return actor_loss.item(), (critic1_loss.item() + critic2_loss.item()) / 2

    def _soft_update_targets(self, tau=0.005):
        """Soft update of target network parameters."""
        for target_param, param in zip(self.critic_target1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic_target2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self, num_episode):
        print(f"... Save Model to {self.save_path}")
        torch.save({
            "episode": int(num_episode),
            "actor" : self.actor.state_dict(),
            "critic1" : self.critic1.state_dict(),
            "critic2" : self.critic2.state_dict(),
            "optimizer" : self.actor_optimizer.state_dict(),
            "lr": float(self.actor_optimizer.param_groups[0]['lr'])
        }, self.save_path+'/ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
