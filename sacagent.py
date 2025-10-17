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


class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # N-step learning parameter, initialized to 1
        self.n_step = 1

        # Replay Memory
        self.min_replay_memory_size = 1000
        self.replay_memory = deque(maxlen=100000)
        self.batch_size = 32

        # Device selection
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"

        # Network dimensions
        model_state_dim = (self.batch_size,) + self.state_dim
        model_action_dim = (self.batch_size, self.action_dim)

        # Networks
        self.actor = ActorNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.critic1 = CriticNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.critic2 = CriticNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.critic_target1 = CriticNetwork(model_state_dim, model_action_dim).float().to(self.device)
        self.critic_target2 = CriticNetwork(model_state_dim, model_action_dim).float().to(self.device)

        # Initialize target networks with critic network weights
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

    def update_replay_memory(self, state, action, reward, next_state, done):
        """Save experience to replay buffer.
    
        States are stored as uint8 numpy arrays.
        Action, reward, and done are stored as python native types.
        """
        # The LazyFrame from the environment wraps uint8 arrays.
        # We convert it to a numpy array with the correct dtype.
        state_arr = np.array(state, dtype=np.uint8)
        next_state_arr = np.array(next_state, dtype=np.uint8)
        
        self.replay_memory.append((state_arr, action, reward, next_state_arr, done))

    def _sample_experience(self):
        """
        Samples a batch of experience from the replay buffer.
        This is where N-step logic will be implemented.
        For now, with n_step=1, it performs standard random sampling.
        """
        if self.n_step == 1:
            # Return a batch of experiences (as python native types and numpy arrays)
            return random.sample(self.replay_memory, self.batch_size)
        else:
            # N-step sampling logic will be implemented here later
            raise NotImplementedError(f"N-step sampling for n_step={self.n_step} is not implemented yet.")

    def act(self, state, training=True):
        """
        Selects an action for a single state.
        During the initial exploration phase, it returns a random action.
        Otherwise, it uses the policy network to sample an action.
        """
        # Take random actions until the replay buffer has collected a minimum number of experiences
        if len(self.replay_memory) < self.min_replay_memory_size:
            return random.randrange(self.action_dim)

        # state is a LazyFrame or np.array
        state_arr = np.array(state, dtype=np.uint8)
        # Add a batch dimension, convert to float tensor, normalize, and send to device
        state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

        # Set the network to evaluation or training mode and get action probabilities
        self.actor.train(training)
        with torch.no_grad():
            pi = self.actor(state_tensor)

            # Use Categorical distribution for sampling
            dist = torch.distributions.Categorical(probs=pi)
            action = dist.sample()

        # Return the action as a Python integer
        return action.item()