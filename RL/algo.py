from abc import abstractmethod
from abc import ABC
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pfrl import replay_buffers


class Algorithm(ABC):
    def explore(self, state):  # 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す.
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.tensor(state, dtype=torch.float, device=dev).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state, False)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):  # 決定論的な行動を返す
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.tensor(state, dtype=torch.float, device=dev).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor.sample(state, True)
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, steps):  # 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す.
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
        """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buf = replay_buffers.ReplayBuffer(capacity=buffer_size)
        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def append(self, state, action, reward, done, next_state):
        state_ = torch.from_numpy(state)
        act_ = torch.from_numpy(action)
        n_state_ = torch.from_numpy(next_state)
        self.buf.append(state_, act_, torch.Tensor([reward]), n_state_, is_state_terminal=done)

    def sample(self, batch_size):
        states, acts, rews, dones, n_states = [], [], [], [], []
        for obs in self.buf.sample(batch_size):
            states.append(obs[0]["state"])
            acts.append(obs[0]["action"])
            rews.append(obs[0]["reward"])
            dones.append(torch.Tensor([float(obs[0]["is_state_terminal"])]))
            n_states.append(obs[0]["next_state"])
        states = torch.cat(states).reshape(len(states), *states[0].shape).to(self.dev)
        n_states = torch.cat(n_states).reshape(len(n_states), *n_states[0].shape).to(self.dev)
        acts = torch.cat(acts).reshape(len(acts), *acts[0].shape).to(self.dev)
        rews = torch.cat(rews).reshape(len(rews), *rews[0].shape).to(self.dev)
        dones = torch.cat(dones).reshape(len(dones), *dones[0].shape).to(self.dev)
        ans = (states, acts, rews, dones, n_states)
        return ans


class Trainer:
    def __init__(self, env, algo, seed=0, num_steps=10 ** 8, eval_interval=10 ** 2, num_eval_episodes=1):
        self.env = env
        self.env_test = env
        self.algo = algo
        # 環境の乱数シードを設定する．
        self.env.seed(seed)

        self.returns = {'step': [], 'return': []}  # 平均収益を保存するための辞書．
        self.num_steps = num_steps  # データ収集を行うステップ数．
        self.eval_interval = eval_interval  # 評価の間のステップ数(インターバル)．
        self.num_eval_episodes = num_eval_episodes  # 評価を行うエピソード数．
        self.eval_id = 0  # 現在のevalの番号

    def train(self):  # num_stepsステップの間，データ収集・学習・評価を繰り返す．
        self.start_time = time()  # 学習開始の時間
        writer = SummaryWriter(log_dir="./logs")
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        writer.add_graph(self.algo.actor, torch.from_numpy(np.zeros(shape=(1,32*4))).float().to(dev))
        writer.add_graph(self.algo.critic, (torch.from_numpy(np.zeros(shape=(1, 32*4))).float().to(dev),
                         torch.from_numpy(np.zeros(shape=(1, 2))).float().to(dev)))

        t = 0  # エピソードのステップ数．
        state = self.env.reset()  # 環境を初期化する．
        for steps in tqdm(range(1, self.num_steps + 1)):
            # 環境(self.env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            # アルゴリズムに渡し，状態・エピソードのステップ数を更新する．
            state, t = self.algo.step(self.env, state, t, steps)
            if self.algo.is_update(steps):  # アルゴリズムが準備できていれば，1回学習を行う．
                l_a1, l_c1, l_c2 = self.algo.update()
                writer.add_scalar("actor loss", l_a1, steps)
                writer.add_scalar("critic loss1", l_c1, steps)
                writer.add_scalar("critic loss2", l_c2, steps)
            if steps % self.eval_interval == 0:  # 一定のインターバルで評価する．
                rew_ave = self.evaluate(steps)
                writer.add_scalar("evaluate rew", rew_ave, steps)
                torch.save(self.algo.actor.cpu().state_dict(), './models/actor.pth')
                self.algo.actor.to(dev)
                torch.save(self.algo.critic.cpu().state_dict(), './models/critic.pth')
                self.algo.critic.to(dev)
                torch.save(self.algo.critic_target.cpu().state_dict(), './models/c_target.pth')
                self.algo.critic_target.to(dev)
        writer.close()

    def evaluate(self, steps):  # 複数エピソード環境を動かし，平均収益を記録する．
        returns = []
        ave_rew = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            episode_return = 0.0
            while (not done):
                action = self.algo.exploit(state)
                # print(" eval action {}".format(action))
                state, reward, done, _ = self.env_test.step(action, True)
                episode_return += reward
            ave_rew += episode_return
            returns.append(episode_return)
        ave_rew /= self.num_eval_episodes
        mean_return = np.mean(returns)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')
        self.env_test.generate_mp4()
        return ave_rew

    def plot(self):
        """ 平均収益のグラフを描画する． """
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Steps', fontsize=24)
        plt.ylabel('Return', fontsize=24)
        plt.tick_params(labelsize=18)
        plt.title(f'{self.env.unwrapped.spec.id}', fontsize=24)
        plt.tight_layout()

    @property
    def time(self):
        """ 学習開始からの経過時間． """
        return str(timedelta(seconds=int(time() - self.start_time)))


def calc_log_pi(stds, noises, actions):
    #  calc : \log\pi(a|s) = \log p(u|s) - \sum_{i=1}^{|\mathcal{A}|} \log (1 - \tanh^{2}(u_i))
    #  これは, \epsilon * \sigma ~ N(0, \sigma)なる確率密度の対数を計算する関数.
    # act = tanh(\mu + \epsilon*\sigma) より, log \pi(a|s) = log p(u|s) - log (1 - tanh'(u)),  (u = \mu + \epsilon*\sigma)
    gaussian_log_probs = torch.distributions.Normal(torch.zeros_like(stds), stds).log_prob(noises).sum(dim=-1,
                                                                                                       keepdim=True)
    log_pis = gaussian_log_probs - torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis


def reparameterize(means, log_stds):
    # acts ~ N(means, stds), log_pis = f(acts), f:N(means, stds)
    stds = log_stds.exp()
    noises = stds * torch.randn_like(means)
    tmp = noises + means  # tmp ~ N(means, stds)
    acts = torch.tanh(tmp)
    log_pis = calc_log_pi(stds=stds, noises=noises, actions=acts)
    return acts, log_pis



