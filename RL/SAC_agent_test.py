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
from gym.envs.registration import register

register(
    id='myenv-v2',
    entry_point='env:MyEnv2'
)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


def init_weights(m):
    if (type(m) is nn.Conv2d) or (type(m) is nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Q_Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net1 = nn.Sequential(
            nn.Linear(state_shape + action_shape[0], num),
            nn.ReLU(inplace=True),
            # nn.Linear(num, num),
            # nn.ReLU(inplace=True),
            # nn.Linear(num, num),
            # nn.ReLU(inplace=True),
            nn.Linear(num, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.net1.apply(init_weights)

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.net1(inputs)


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
    # env = gym.make("myenv-v2", conf=conf)
    return env


def make_policy(state_shape, action_shape):
    # net = nn.Sequential(
    #     nn.Conv2d(3, num, 4, stride=2),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(num, 64, 4, stride=2),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(64, 32, 4, stride=2),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(32, 8, 4, stride=2),
    #     nn.ReLU(inplace=True),
    #     Flatten(),  # torch.Size([1, 192])
    #     nn.Linear(192, 64),  # torch.Size([1, 64])
    #     nn.ReLU(inplace=True),
    #     nn.Linear(64, 2 * 2),
    #     Lambda(squashed_diagonal_gaussian_head),
    # )
    num = 256
    net = nn.Sequential(
        # nn.Linear(state_shape[0], num),
        nn.Linear(state_shape, num),
        nn.ReLU(inplace=True),
        # nn.Linear(num, num),
        # nn.ReLU(inplace=True),
        # nn.Linear(num, num),
        # nn.ReLU(inplace=True),
        nn.Linear(num, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 2 * action_shape[0]),
        Lambda(squashed_diagonal_gaussian_head),
    )
    net.apply(init_weights)
    return net


def train_PFRL_agent():
    env = make_env()
    env = pfrl.wrappers.CastObservationToFloat32(env)
    env = pfrl.wrappers.NormalizeActionSpace(env)

    policy = make_policy(env.state_shape, env.action_space.shape).to(dev)
    q_func1 = Q_Net(env.state_shape, env.action_space.shape).to(dev)
    q_func2 = Q_Net(env.state_shape, env.action_space.shape).to(dev)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    q_func1_optimizer = torch.optim.Adam(q_func1.parameters(), lr=3e-4)
    q_func2_optimizer = torch.optim.Adam(q_func2.parameters(), lr=3e-4)

    gamma = 0.99
    gpu = -1
    replay_start_size = 5 * 10 ** 1
    # replay_start_size = 0
    minibatch_size = 1
    max_grad_norm = 0.5
    update_interval = 1
    replay_buffer = replay_buffers.ReplayBuffer(10**6)

    def burn_in_action_func():
        """Select random actions until model is updated one or more times."""
        print("burn in action func")
        ans = np.random.uniform(-1, 1, size=2).astype(np.float32)
        print("ans shape {}, ans type {}".format(ans.shape, type(ans)))
        return ans

    print(torch.cuda.is_available())
    agent = pfrl.agents.SoftActorCritic(policy,
                                        q_func1, q_func2, policy_optimizer, q_func1_optimizer,
                                        q_func2_optimizer, replay_buffer, gamma, gpu, replay_start_size,
                                        minibatch_size, update_interval, max_grad_norm, temperature_optimizer_lr=3e-4,
                                        burnin_action_func=burn_in_action_func)
    eval_interval = 2 * 10 ** 1
    policy_start_step = 5 * 10 ** 1

    # experiments.train_agent_with_evaluation(
    #     agent=agent,
    #     env=env,
    #     steps=3*10**6,
    #     eval_n_steps=100,
    #     eval_n_episodes=None,
    #     eval_interval=1,
    #     outdir="./",
    #     save_best_so_far_agent=True,
    #     eval_env=env,
    # )
    state = env.reset()
    state_test = state.reshape(1, -1)
    print("state shape {}, type {}".format(state.shape, type(state)))
    print("state_test shape {}, type {}".format(state_test.shape, type(state_test)))

    with agent.eval_mode():
        agent.act(state_test)
    for i in tqdm(range(3*10**6)):
        if i // eval_interval == 0 and i != 0:
            with agent.eval_mode():
                state = env.reset()
                state = torch.from_numpy(state).to(dev)
                r_sum = 0
                while True:
                    act = agent.act(state)
                    n_state, rew, done, info = env.step(act)
                    r_sum += rew
                    state = torch.from_numpy(n_state).to(dev)
                    if done:
                        print("step {}: rew is {}.".format(i, r_sum))
                        state = env.reset()
                        break
        # if i < policy_start_step:
        #     act = env.action_space.sample()
        # else:
        act = agent.act(state)
        print("act {}".format(act))
        n_state, rew, done, info = env.step(act)
        agent.observe(n_state, rew, done, done)
        if done:
            state = env.reset()


if __name__ == "__main__":
    train_PFRL_agent()