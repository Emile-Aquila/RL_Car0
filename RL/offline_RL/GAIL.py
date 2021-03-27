import torch
import numpy as np
from offline_RL.models import Discriminator
from offline_RL.PPO import PPO
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GAIL(PPO):
    def __init__(self, buffer_exp, state_shape, action_shape, device=dev, seed=0,
                 batch_size=50000, batch_size_disc=64, lr=3e-4, gamma=0.995,  rollout_length=50000,
                 epoch_disc=10, epoch_ppo=50, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, lr, gamma,  rollout_length,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm)

        # デモンストレーションデータの保持するバッファ．
        self.buffer_exp = buffer_exp

        # Discriminator．
        self.disc = Discriminator(state_shape, action_shape).to(device)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(), lr=lr)

        self.batch_size_disc = batch_size_disc
        self.epoch_disc = epoch_disc

    def update(self):
        # GAILでは，環境からの報酬情報は用いない．
        states, actions, _, dones, log_pis = self.buffer.get()
        # Discriminatorの学習．
        for _ in range(self.epoch_disc):
            idxes = np.random.randint(low=0, high=self.rollout_length, size=self.batch_size_disc)
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size_disc)[:2]
            self.update_disc(states[idxes], actions[idxes], states_exp, actions_exp)
        with torch.no_grad():
            rewards = - torch.log(torch.nn.functional.sigmoid(self.disc(states[:-1], actions)))
        # PPOの学習．
        self.update_ppo(states, actions, rewards, dones, log_pis)

    def update_disc(self, states, actions, states_exp, actions_exp):
        loss_pi = -torch.log(torch.nn.functional.sigmoid(self.disc(states, actions))).mean()
        loss_exp = -torch.log(torch.nn.functional.sigmoid(-self.disc(states_exp, actions_exp))).mean()
        self.optim_disc.zero_grad()
        (loss_pi + loss_exp).backward()
        self.optim_disc.step()
