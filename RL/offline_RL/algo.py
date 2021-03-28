import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SerializedBuffer:
    def __init__(self, path, device=dev):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device
        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):
    def __init__(self, buffer_size, state_shape, action_shape, device=dev):
        self._p = 0
        self._n = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty((buffer_size, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device=dev):
        self.states = torch.empty((buffer_size + 1, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)

        self._p = 0
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self._p = (self._p + 1) % self.buffer_size

    def append_last_state(self, last_state):
        assert self._p == 0, 'Buffer needs to be full before appending last_state.'
        self.states[self.buffer_size].copy_(torch.from_numpy(last_state))

    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis


def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(
        2 * math.pi) * log_stds.size(-1)
    # tanh による確率密度の変化を修正する．
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)
    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)
    return actions, log_pis


def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    """ 平均(mean)，標準偏差の対数(log_stds)でパラメータ化した方策における，行動(actions)の確率密度の対数を計算する． """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def calculate_advantage(values, rewards, dones, gamma=0.995, lambd=0.997):
    """ GAEを用いて，状態価値のターゲットとGAEを計算する． """

    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    gaes = torch.empty_like(rewards)

    gaes[-1].copy_(deltas[-1])
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t].copy_(deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1])

    targets = gaes + values[:-1]
    return targets, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def add_random_noise(action, std):
    """ データ収集時に，行動にガウスノイズを載せる． """
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_data(sac_agent, env, weight_path, buffer_size, std=0.0, p_rand=0.0, device=dev, seed=0):
    # # 環境を構築する．
    # 学習済みの重みを読み込む．
    # sac_agent.actor.load(weight_path)
    sac_agent.load_state_dict(torch.load(weight_path))
    sac_agent.to(dev)
    # シードを設定する．
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # GPU上にリプレイバッファを作成する．
    print("device {}".format(device))
    buffer = Buffer(buffer_size, env.state_shape, env.action_space.shape, device)
    # エキスパートの平均収益を記録する．
    total_return = 0.0
    num_episodes = 0
    # 環境を初期化する．
    state = env.reset()
    t = 0
    episode_return = 0.0
    for steps in tqdm(range(1, buffer_size + 1)):
        t += 1
        # ノイズを加えつつ，行動選択する．
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            # action = sac_agent(torch.tensor(state, dtype=torch.float, device=device).unsqueeze_(0)).cpu().numpy()[0]
            act, log_std = sac_agent(torch.tensor(state, dtype=torch.float, device=device))
            action = act.detach().cpu().numpy()
            std = log_std.exp().detach().cpu().numpy()
            action = add_random_noise(action, std)
        # 環境を1ステップ進める．
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先もMDPが継続するはず．よって，終了シグナルをFalseにする．
        # mask = False if t == env._max_episode_steps else done
        # リプレイバッファにデータを追加する．
        buffer.append(state, action, reward, done, next_state)
        # エピソードが終了した場合には，環境をリセットする．
        if done:
            total_return += episode_return
            num_episodes += 1
            state = env.reset()
            t = 0
            episode_return = 0.0
        state = next_state
    if num_episodes > 0:
        print(f'Mean return of the expert is {total_return / num_episodes:.2f}．')
    return buffer