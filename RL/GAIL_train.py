import gym
from offline_RL.GAIL import GAIL
from offline_RL.algo import SerializedBuffer, collect_data
from offline_RL.trainer import GAILTrainer
import gym_donkeycar
from SAC_model import ActorNetwork
from SAC import SAC
from env import MyEnv
import os

exe_path = f"/home/emile/.local/lib/python3.9/site-packages/gym_donkeycar/DonkeySimLinux/donkey_sim.x86_64"
conf = {"exe_path": exe_path, "port": 9091}
env = gym.make("donkey-generated-track-v0", conf=conf)
env = MyEnv(env)
env_online = env


# デモンストレーション時のガウスノイズの標準偏差．
STD = 0.0
# デモンストレーション時にランダムに行動する確率．
P_RAND = 0.05
BUFFER_SIZE = 10 ** 5
# BUFFER_SIZE = 10 ** 2
SEED = 0

# 学習済みの重みのパス．
WEIGHT_OPTIMAL = os.path.join("./model_for_gail/actor.pth")
# バッファを保存するパス．
BUFFER_OPTIMAL = os.path.join("./" + f'buffer_optimal_std{STD}_p_rand{P_RAND}.pth')

# sac_agent = SAC(
#     state_shape=env.state_shape,
#     # state_shape=32,
#     action_shape=env.action_space.shape,
#     seed=SEED,
#     reward_scale=1.0,
#     # start_steps=5 * 10,
#     start_steps=0,
#     batch_size=0,
#     buffer_size=1,
# )
agent = ActorNetwork(state_shape=env.state_shape, action_shape=env.action_space.shape)


buffer = collect_data(agent, env, WEIGHT_OPTIMAL, BUFFER_SIZE, STD, P_RAND)
buffer.save(BUFFER_OPTIMAL)

# PPOの学習バッチサイズ．
BATCH_SIZE = 1000
# BATCH_SIZE = 1000
# ロールアウトの長さ．
ROLLOUT_LENGTH = 300
# ロールアウト毎のPPOの学習エポック数．
EPOCH_DISC = 10
# ロールアウト毎のDiscriminatorの学習エポック数．
EPOCH_PPO = 200

NUM_STEPS = 50000
EVAL_INTERVAL = 500


algo = GAIL(
    buffer_exp=SerializedBuffer(BUFFER_OPTIMAL),
    state_shape=env.state_shape,
    action_shape=env.action_space.shape,
    seed=SEED,
    batch_size=BATCH_SIZE,
    rollout_length=ROLLOUT_LENGTH,
    epoch_disc=EPOCH_DISC,
    epoch_ppo=EPOCH_PPO
)
trainer = GAILTrainer(
    env=env,
    env_online=env_online,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL
)
trainer.train()