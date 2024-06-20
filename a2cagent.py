import datetime

import torch.cuda
import torch.nn.functional as F

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from einops.layers.torch import Rearrange


class A2CModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.model_common = self.__build_cnn(c)

        self.pi = nn.Linear(512, output_dim)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = self.model_common(x)

        return F.softmax(self.pi(x)), self.v(x)

    def __build_cnn(self, c):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Rearrange('c h w -> (c h w)'),
            nn.Linear(3136, 512),
            nn.ReLU()
        )


class A2CAgent:
    def __init__(self, state_dim, action_dim, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        ## hyper parameters
        self.discount_factor = 0.9
        date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_path = f"./saved_a2c/{date_time}"
        self.load_path = f"./saved_a2c/~~"
        ##

        self.use_cuda = torch.cuda.is_available()

        self.model = A2CModel(self.state_dim, self.action_dim).float()
        self.device = "cpu"
        if self.use_cuda:
            self.model = self.model.to(device='cuda')
            self.device = "cuda"

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00025)

        self.writer = SummaryWriter(self.save_path)

    def act(self, state, training=True):
        self.model.train(training)

        pi, _ = self.model(torch.FloatTensor(state).to(self.device))
        action = torch.multinomial(pi, num_samples=1).cpu().numpy()[0]
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])

        pi, value = self.model(state)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            target_value = reward + (1-done) * self.discount_factor * next_value

        critic_loss = F.mse_loss(value, target_value)

        eye = torch.eye(self.action_dim).to(self.device)
        one_hot_action = eye[action.view(-1).long()]

        advantage = (target_value - value).detach()
        actor_loss = -(torch.log((one_hot_action * pi).sum(1))*advantage).mean()

        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_model(self):
        print(f"... Save Model to {self.save_path}")
        torch.save({
            "network" : self.model.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, self.save_path+'/ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)
