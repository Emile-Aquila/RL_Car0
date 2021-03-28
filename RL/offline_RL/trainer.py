import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
from datetime import timedelta
from time import time
from torch.utils.tensorboard import SummaryWriter
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trainer:
    def __init__(self, env, algo, seed=0, num_steps=3 * 10 ** 4, eval_interval=10 ** 3, num_eval_episodes=5):
        # 評価用の環境．
        self.env = env
        self.env.seed(seed)

        # 学習アルゴリズム．
        self.algo = algo
        # 平均収益を保存するための辞書．
        self.returns = {'step': [], 'return': []}
        # 学習ステップ数．
        self.num_steps = num_steps
        # 評価のインターバル．
        self.eval_interval = eval_interval
        # 評価を行うエピソード数．
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # 学習開始の時間
        self.start_time = time()
        for step in range(1, self.num_steps + 1):
            self.algo.update()
            if step % self.eval_interval == 0:
                self.evaluate(step)

    def evaluate(self, step):
        """ 複数エピソード環境を動かし，平均収益を記録する． """
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.algo.select_action(state)
                state, reward, done, _ = self.env.step(action)
                total_return += reward

        mean_return = total_return / self.num_eval_episodes
        self.returns['step'].append(step)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')
        return mean_return

    # def visualize(self):
    #     """ 1エピソード環境を動かし，mp4を再生する． """
    #     env = gym.envs.wrap_monitor(gym.make(self.env.unwrapped.spec.id))
    #     state = env.reset()
    #     done = False
    #
    #     while (not done):
    #         action = self.algo.select_action(state)
    #         state, _, done, _ = env.step(action)
    #
    #     del env
    #     return play_mp4()

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


class GAILTrainer(Trainer):
    def __init__(self, env, env_online, algo, seed=0, num_steps=3 * 10 ** 4, eval_interval=10 ** 3,
                 num_eval_episodes=5):
        super().__init__(env, algo, seed, num_steps, eval_interval, num_eval_episodes)

        # データ収集用の環境．
        self.env_online = env_online
        self.env_online.seed(2 ** 20 - seed)

    def train(self):
        # 学習開始の時間
        self.start_time = time()
        writer = SummaryWriter(log_dir="./logs")
        # エピソードのステップ数．
        t = 0
        # 環境を初期化する．
        state = self.env_online.reset()
        eval_time = 0
        max_mean = None
        for step in range(1, self.num_steps + 1):
            # 環境を1ステップ進める．
            state, t = self.algo.step(self.env_online, state, t, step)

            # 一定のインターバルで学習する．
            if self.algo.is_update(step):
                l_c, l_a = self.algo.update()
                writer.add_scalar("actor loss", l_a, t)
                writer.add_scalar("critic loss", l_c, t)

            # 一定のインターバルで評価する．
            if step % self.eval_interval == 0:
                mean_return = self.evaluate(step)
                writer.add_scalar("average rew", mean_return, eval_time)
                eval_time += 1
                if max_mean is None or mean_return > max_mean:
                    max_mean = mean_return
                    torch.save(self.algo.actor.cpu().state_dict(), './models_GAIL/actor.pth')
                    self.algo.actor.to(dev)
                    torch.save(self.algo.critic.cpu().state_dict(), './models_GAIL/critic.pth')
                    self.algo.critic.to(dev)
