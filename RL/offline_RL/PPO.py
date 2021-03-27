import torch
import torch.nn as nn
import numpy as np
from offline_RL.algo import RolloutBuffer
from offline_RL.models import PPOActor, PPOCritic
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calculate_advantage(values, rewards, dones, gamma=0.995, lambd=0.997):
    """ GAEを用いて，状態価値のターゲットとGAEを計算する． """

    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    gaes = torch.empty_like(rewards)

    gaes[-1].copy_(deltas[-1])
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t].copy_(deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1])

    targets = gaes + values[:-1]
    return targets, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO:
    def __init__(self, state_shape, action_shape, device=dev, seed=0,
                 batch_size=2048, lr=3e-4, gamma=0.995, rollout_length=2048, epoch_ppo=50,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.):

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # データ保存用のバッファ．
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        self.actor = PPOActor(state_shape, action_shape).to(device)
        self.critic = PPOCritic(state_shape).to(device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # その他パラメータ．
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        # ゲームオーバーではなく，最大ステップ数に到達したことでエピソードが終了した場合は，
        # 本来であればその先もMDPが継続するはず．よって，終了シグナルをFalseにする．
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi)
        # ロールアウトの終端に達したら，最終状態をバッファに追加する．
        if step % self.rollout_length == 0:
            self.buffer.append_last_state(next_state)

        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        states, actions, rewards, dones, log_pis = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis)

    def update_ppo(self, states, actions, rewards, dones, log_pis):
        with torch.no_grad():
            values = self.critic(states)
        # GAEを計算する．
        targets, advantages = calculate_advantage(values, rewards, dones, self.gamma, self.lambd)
        # PPOを更新する．
        for _ in range(self.epoch_ppo):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start + self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
