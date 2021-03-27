import torch
import torch.nn as nn
import numpy as np
from offline_RL.algo import reparameterize, evaluate_lop_pi


class PPOActor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            nn.Linear(state_shape, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class PPOCritic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            nn.Linear(state_shape, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 1),
        )

    def forward(self, states):
        return self.net(states)


class Discriminator(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], num),
            nn.Tanh(),
            nn.Linear(num, 2*num),
            nn.Tanh(),
            nn.Linear(2*num, num),
            nn.Tanh(),
            nn.Linear(num, num),
            nn.Tanh(),
            nn.Linear(num, 1)
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=1))
