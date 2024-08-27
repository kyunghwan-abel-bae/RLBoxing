import datetime
import random

import torch.cuda
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from einops.layers.torch import Rearrange

from collections import deque


class A2CModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        b, c, h, w = input_dim
        b, a = output_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.model_common = self.__build_cnn(c)

        self.pi = nn.Linear(512, a)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        # print(f"x : {x}")
        x = self.model_common(x)
        # print(f"model_commonx : {x}")

        a = self.pi(x)
        b = self.v(x)
        # print("a, b done")
        # print(a)

        return F.softmax(a), b#F.softmax(self.pi(x)), self.v(x)
        # return F.softmax(self.pi(x)), self.v(x)

    def __build_cnn(self, c):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(3136, 512),
            nn.ReLU()
        )


class A2CAgent:
    def __init__(self, state_dim, action_dim, checkpoint=None, func_print=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        ## hyper parameters
        self.discount_factor = 0.9
        date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = f"./saved_a2c/{date_time}"
        self.load_path = checkpoint#f"./saved_a2c/20240705114408/ckpt"
        ##

        ## added
        self.min_replay_memory_size = 500
        self.replay_memory = deque(maxlen=10000)
        self.batch_size = 16
        ## added

        self.use_cuda = torch.cuda.is_available()

        model_state_dim = (self.batch_size,) + self.state_dim
        model_action_dim = (self.batch_size, self.action_dim)
        self.model = A2CModel(model_state_dim, model_action_dim).float()

        self.device = "cpu"
        if self.use_cuda:
            self.model = self.model.to(device='cuda')
            self.device = "cuda"

        self.init_lr = 1e-6
        self.min_lr = 1e-7
        #
        self.func_print = func_print

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)

        self.data_load = None

        # LOAD MODEL
        if checkpoint:
            self.data_load = torch.load(self.load_path, map_location=('cuda' if self.use_cuda else 'cpu'))

            self.init_lr = self.data_load.get("lr")

            self.model.load_state_dict(self.data_load.get('network'))
            self.optimizer.load_state_dict(self.data_load.get('optimizer'))

        # lambda_lr = lambda epoch: max(self.min_lr, self.init_lr*(0.995 ** epoch))
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=self.min_lr)

        self.writer = SummaryWriter(self.save_path)

        # for Test
        self.pi_counter = 0

    def update_replay_memory(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        # print("action : ", action)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.FloatTensor([reward]).cuda() if self.use_cuda else torch.FloatTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.replay_memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        # if len(self.replay_memory) < self.min_replay_memory_size:
        #     return random.randint(0, self.action_dim-1)

        self.model.train(training)

        pi, _ = self.model(torch.FloatTensor(state).to(self.device))

        self.pi_counter += 1
        if self.pi_counter % 500 == 0:
            self.func_print(f"pi : {pi}")
            self.pi_counter = 0

        # if torch.isnan(pi).any() or torch.isinf(pi).any() or (pi < 0).any():
        #     print("Invalid values in probability tensor:", pi)
        # print(f"pi.shape : {pi.shape}")
        # print(f"pi : {pi}")

        action = torch.multinomial(pi, num_samples=1).cpu().numpy()[0]

        # print(f"action in act : {action}")
        action = action[0]

        # print(f"action in act : {action}")
        return action

    # def learn(self, state, action, reward, next_state, done):
    def learn(self):#, state, action, reward, next_state, done):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return -1, -1

        samples = random.sample(self.replay_memory, self.batch_size)

        state, action, reward, next_state, done = map(torch.stack, zip(*samples))

        pi, value = self.model(state)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            target_value = reward + (1-done.float()) * self.discount_factor * next_value

        critic_loss = F.mse_loss(value, target_value)

        eye = torch.eye(self.action_dim).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        one_hot_action = one_hot_action.view(action.size(0), -1, self.action_dim)
        advantage = (target_value - value).detach()

        actor_loss = -(torch.log((one_hot_action * pi).sum(2)) * advantage).mean()

        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_model(self, num_episode):
        print(f"... Save Model to {self.save_path}")
        torch.save({
            "episode": int(num_episode),
            "network" : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "lr": float(self.optimizer.param_groups[0]['lr'])
        }, self.save_path+'/ckpt')

    # def load_model(self, load_path):
    #     print("implementing")
    #     if not load_path.exists():
    #         raise ValueError(f"{load_path} does not exist")

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


