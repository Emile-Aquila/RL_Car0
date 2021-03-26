import numpy as np
import torch
from SAC_model import ActorNetwork, CriticNetwork
from algo import Algorithm, ReplayBuffer
from PIL import Image
import gym_donkeycar
import gym
import random


def show_state(state_np):
    state_np_ = state_np[50:130, 0:160, :]
    # state_np_ = state_np
    img = Image.fromarray(state_np_, "RGB").convert("L")
    # img = Image.fromarray(state_np_, "RGB").convert("L").point(lambda x: 0 if x < 190 else x)
    print(img)
    frame = np.array(img, dtype=np.float32)
    print(frame.shape)
    img.show()


class SAC(Algorithm):
    def __init__(self, state_shape, action_shape, seed=0, batch_size=256, gamma=0.99, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, buffer_size=5 * 10 ** 3, start_steps=5 * 10 ** 3, tau=5e-3,
                 min_alpha=0.3, reward_scale=1.0, epsilon=0.00):
        super().__init__()

        self.dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
        )
        self.actor = ActorNetwork(state_shape, action_shape).to(self.dev)
        self.critic = CriticNetwork(state_shape, action_shape).to(self.dev)

        # Target Network を用いて学習を安定化させる.
        self.critic_target = CriticNetwork(state_shape, action_shape).to(self.dev).eval()

        # adjust entropy(\alpha)
        self.min_alpha = torch.tensor(min_alpha)
        self.alpha = torch.tensor(1.0, requires_grad=True)

        # init target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_alpha = torch.optim.Adam([self.alpha], lr=lr_alpha)

        # param
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.learning_steps = 0
        self.total_rew = 0.0
        self.epsilon = epsilon

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps):
        t += 1
        if steps <= self.start_steps or (self.epsilon >= random.random()):  # 最初はランダム.
            action = env.action_space.sample()
            action[1] = action[1] * 2.0 - 1.0
        else:
            action, _ = self.explore(state)
        n_state, rew, done, info = env.step(action)
        self.total_rew += rew
        self.buffer.append(state, action, rew, done, n_state)  # add data to buffer
        if done:  # エピソードが終了した場合には，環境をリセットする．
            t = 0
            n_state = env.reset()
            self.total_rew = 0.0
        return n_state, t

    def actor_loss_func(self, states):
        acts, log_pis = self.actor.sample(states)
        q1, q2 = self.critic(states, acts)
        loss_actor = (self.alpha * log_pis - torch.min(q1, q2)).mean()
        return loss_actor, log_pis

    def critic_loss_func(self, states, actions, rews, dones, n_states):
        now_q1, now_q2 = self.critic(states, actions)
        with torch.no_grad():
            n_actions, log_pis = self.actor.sample(n_states, False)
            q1, q2 = self.critic_target(n_states, n_actions)
            target_vs = torch.min(q1, q2) - self.alpha * log_pis

        target_qs = self.reward_scale * rews + self.gamma * target_vs * (1.0 - dones)  # r(s,a) + \gamma V(s')
        # loss funcs
        loss_c1 = (now_q1 - target_qs).pow_(2).mean()
        loss_c2 = (now_q2 - target_qs).pow_(2).mean()
        return loss_c1, loss_c2

    def update_critic(self, states, actions, rews, dones, n_states):
        # (r(s,a) + \gamma V(s') - Q(s,a))^2 = (r(s,a) + \gamma {min[Q(s',a')] - \alpha \log \pi (a|s)} - Q(s,a))^2
        loss_c1, loss_c2 = self.critic_loss_func(states, actions, rews, dones, n_states)
        # update
        self.optim_critic.zero_grad()
        (loss_c1 + loss_c2).backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optim_critic.step()
        return loss_c1.clone().detach(), loss_c2.clone().detach()

    def update_actor(self, states):
        loss_actor, log_pis = self.actor_loss_func(states)
        # update
        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optim_actor.step()
        self.entropy_adjust_func(log_pis)
        return loss_actor.clone().detach(), log_pis.clone().detach()

    def update_target(self):
        for target, trained in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.mul_(1.0 - self.tau)
            target.data.add_(self.tau * trained.data)

    def update(self):
        self.learning_steps += 1
        states, actions, rews, dones, n_states = self.buffer.sample(self.batch_size)
        l_c1, l_c2 = self.update_critic(states, actions, rews, dones, n_states)
        l_a1, log_p = self.update_actor(states)
        self.update_target()
        return l_a1, log_p, l_c1, l_c2

    def entropy_adjust_func(self, log_pis):
        with torch.no_grad():
            loss = log_pis + self.min_alpha
        loss = -(self.alpha * loss).mean()
        self.optim_alpha.zero_grad()
        loss.backward()
        self.optim_alpha.step()


def main():
    exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
    conf = {"exe_path": exe_path, "port": 9091}
    env = gym.make("donkey-generated-track-v0", conf=conf)
    state = env.reset()
    show_state(np.array(state))


if __name__ == "__main__":
    main()
