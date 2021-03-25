import torch
import torch.nn as nn
from tqdm import tqdm
import gym_donkeycar
from env import MyEnv
import gym
import pfrl
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda
from torch import distributions
import numpy as np

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


def init_weights(m):
    if (type(m) is nn.Conv2d) or (type(m) is nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Q_Net(nn.Module):
    def __init__(self):
        super().__init__()
        num = 64
        self.net1 = nn.Sequential(
            nn.Conv2d(3, num, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num, 64, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 8, 4, stride=2),
            nn.ReLU(inplace=True),
            Flatten(),  # torch.Size([1, 192])
            nn.Linear(192, 64),  # torch.Size([1, 64])
            # nn.Linear(64, 2 * 2),
        )
        self.net2 = nn.Sequential(
            nn.Linear(2, 64)
        )
        self.net3 = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net1.apply(init_weights)
        self.net2.apply(init_weights)
        self.net3.apply(init_weights)

    def forward(self, states, acts):
        s1 = self.net1(states)
        a1 = self.net2(acts)
        t1 = torch.cat((s1, a1), dim=-1)
        out1 = self.net3(t1)
        return out1


def squashed_diagonal_gaussian_head(x):
    assert x.shape[-1] == 2 * 2
    mean, log_scale = torch.chunk(x, 2, dim=1)
    log_scale = torch.clamp(log_scale, -20.0, 2.0)
    var = torch.exp(log_scale * 2)
    base_distribution = distributions.Independent(
        distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
    )
    # cache_size=1 is required for numerical stability
    return distributions.transformed_distribution.TransformedDistribution(
        base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
    )


def make_env():
    exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
    conf = {"exe_path": exe_path, "port": 9091}
    env = gym.make("donkey-generated-track-v0", conf=conf)
    env = MyEnv(env)
    return env

def make_policy():
    num = 64
    net = nn.Sequential(
        nn.Conv2d(3, num, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(num, 64, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, 4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 8, 4, stride=2),
        nn.ReLU(inplace=True),
        Flatten(),  # torch.Size([1, 192])
        nn.Linear(192, 64),  # torch.Size([1, 64])
        nn.ReLU(inplace=True),
        nn.Linear(64, 2 * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    net.apply(init_weights)
    return net


def train_PFRL_agent():
    policy = make_policy().to(dev)
    q_func1 = Q_Net().to(dev)
    q_func2 = Q_Net().to(dev)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=3e-4)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=3e-4)
    gamma = 0.99
    gpu = -1
    replay_start_size = 5 * 10 ** 3
    minibatch_size = 256
    max_grad_norm = 0.5
    update_interval = 1
    replay_buffer = replay_buffers.ReplayBuffer(5 * 10 ** 3)

    def burn_in_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(-1, 1, size=2).astype(np.float32)
    print(torch.cuda.is_available())
    agent = pfrl.agents.SoftActorCritic(policy,
                                        q_func1, q_func2, policy_optimizer, q_func1_optimizer,
                                        q_func2_optimizer, replay_buffer, gamma, gpu, replay_start_size,
                                        minibatch_size, update_interval, max_grad_norm, temperature_optimizer_lr=3e-4,
                                        burnin_action_func=burn_in_action_func)
    env = make_env()
    # env1 = make_batch_env(False, env)
    # env2 = make_batch_env(True, env)
    # experiments.train_agent_batch_with_evaluation(
    #     agent=agent,
    #     env=env,
    #     eval_env=env,
    #     outdir="./",
    #     steps=3 * 10 ** 6,
    #     eval_n_steps=None,
    #     eval_n_episodes=2,
    #     eval_interval=2 * 10 ** 3,
    #     log_interval=10,
    #     max_episode_len=None,
    # )
    eval_interval = 2 * 10 ** 1
    policy_start_step = 5 * 10 ** 3
    state = env.reset()
    for i in tqdm(range(3*10**6)):
        if i // eval_interval == 0 and i is not 0:
            with agent.eval_mode():
                state = env.reset()
                r_sum = 0
                while True:
                    act = agent.act(state)
                    n_state, rew, done, info = env.step(act)
                    r_sum += rew
                    if done:
                        print("step {}: rew is {}.".format(i, r_sum))
                        state = env.reset()
                        break
        act = agent.act(state)
        print("act {}".format(act))
        n_state, rew, done, info = env.step(act)
        agent.observe(n_state, rew, done, done)
        if done:
            state = env.reset()


if __name__ == "__main__":
    train_PFRL_agent()