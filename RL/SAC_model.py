import torch
import torch.nn as nn
from algo import reparameterize
import gym_donkeycar
from env import MyEnv
import gym


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


def init_weights(m):
    if (type(m) is nn.Conv2d) or (type(m) is nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            # nn.Linear(state_shape[0], num),
            nn.Linear(state_shape, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(num, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(64, 2 * action_shape[0]),
        )
        self.net.apply(init_weights)

    def forward(self, inputs):
        # calc means and log_stds
        means, log_stds = self.net(inputs).chunk(2, dim=-1)
        return means, log_stds

    def sample(self, inputs, deterministic=False):
        #  select action from inputs
        means, log_stds = self.forward(inputs)
        if deterministic:
            return torch.tanh(means)
        else:
            log_stds = torch.clip(log_stds, -20.0, 2.0)
            return reparameterize(means, log_stds)


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net1 = nn.Sequential(
            nn.Linear(state_shape + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(num, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net1.apply(init_weights)
        self.net2 = nn.Sequential(
            nn.Linear(state_shape + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(num, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net2.apply(init_weights)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=-1)
        return self.net1(inputs), self.net2(inputs)


def main():
    exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
    conf = {"exe_path": exe_path, "port": 9091}
    env = gym.make("donkey-generated-track-v0", conf=conf)
    env = MyEnv(env)
    print("action space {}".format(env.action_space))


if __name__ == "__main__":
    main()
