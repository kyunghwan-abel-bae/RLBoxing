import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SharedEncoder(nn.Module):
    """Actor와 Critic이 공유하는 CNN 특징 추출기"""
    def __init__(self, input_dim):
        super().__init__()
        c, h, w = input_dim[1:]
        if h != 84 or w != 84:
            raise ValueError("Expecting input shape (84, 84)")

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
        )

    def forward(self, x):
        return self.cnn(x)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, encoder):
        super().__init__()
        self.encoder = encoder
        c, h, w = input_dim[1:]
        a = output_dim[1]

        # CNN 출력 크기를 동적으로 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.encoder(dummy_input).shape
            flattened_dim = cnn_out_dim[1] * cnn_out_dim[2] * cnn_out_dim[3]

        # 정책 헤드
        self.head = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flattened_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, a)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, encoder):
        super().__init__()
        self.encoder = encoder
        c, h, w = input_dim[1:]
        a = output_dim[1]

        # CNN 출력 크기를 동적으로 계산
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_out_dim = self.encoder(dummy_input).shape
            flattened_dim = cnn_out_dim[1] * cnn_out_dim[2] * cnn_out_dim[3]

        # Q-Value 헤드
        self.head = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(flattened_dim, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, a)  # 각 행동에 대한 Q-value를 출력
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)