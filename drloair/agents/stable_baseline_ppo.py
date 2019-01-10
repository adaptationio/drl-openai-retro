import time

import gym
import numpy as np
import os
import datetime
import csv
import argparse
from functools import partial

from stable_baselines.common.policies import MlpLnLstmPolicy, FeedForwardPolicy, LstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv,VecNormalize 
from stable_baselines.common import set_global_seeds
from stable_baselines import ACKTR, PPO2
from stable_baselines.deepq import DQN
#from stable_baselines.deepq.policies import FeedForwardPolicy
#from template_env import Template_Gym

#env = Template_Gym()
from sonic_util import AllowBacktracking, make_env
from dumbrain.rl.retro_contest.install_games import installGamesFromDir
installGamesFromDir(romdir='data/roms/')
import retrowrapper
import retro

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')



# Name to identify expert network
expert_name = 'greenhill'
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256, 256],
                                           layer_norm=True,
                                            feature_extraction="mlp")
                    
class CustomPolicy_2(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp",
                                            n_envs=16,
                                            )

class CustomPolicy_3(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy_3, self).__init__(*args, **kwargs,
                                           layers=[256,256,256],
                                           layer_norm=True,
                                            feature_extraction="mlp")


def make_env_2(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = make_env()
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def create_envs():
    env_fns = [] 
    env_names = []
    with open('sonic-train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'Sonic' in row[0]:
                print('Add Environments: ', row[0] + ': ' + row[1])
                env_fns.append(partial(make_env, game=row[0], state=row[1]))
                env_names.append(row[0] + '-' + row[1])

    return env_fns, env_names

#env_id = "CartPole-v1"
#num_cpu = 4  # Number of processes to use
# Create the vectorized environment
#env = DummyVecEnv([lambda: env])
#env = SubprocVecEnv([make_env_2(env_id, i) for i in range(num_cpu)])
#env = SubprocVecEnv([make_env])
num_env = 10
env = SubprocVecEnv([lambda: make_env(stack=False, scale_rew=True)] * num_env)
#env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   #clip_obs=10.)


#env_fns, env_names = create_envs()

#model = PPO2(CnnPolicy, env, learning_rate=1e-5, verbose=1,tensorboard_log="./ppo2fulltime105-3333333333333/" )
model = PPO2(policy=CnnPolicy,
                      env=env,
                      n_steps=8192,
                      nminibatches=8*num_env,
                      lam=0.95,
                      gamma=0.99,
                      noptepochs=4,
                      ent_coef=0.001,
                      learning_rate=lambda _: 2e-5,
                      cliprange=lambda _: 0.2,
                      verbose=1,
                      tensorboard_log="./sonic/")

#model = PPO2(CnnPolicy, env, learning_rate=2e-4, verbose=0,tensorboard_log="trader10year15m-32bk-reward-lr5-10m-2x256")
#model = PPO2(CustomPolicy_2, env, learning_rate=1e-5, verbose=0,tensorboard_log="./ppo2full10-lstm-2/" )
#model = PPO2(MlpLnLstmPolicy, env, verbose=0,tensorboard_log="./ppo2full10-3/" )
#model = PPO2(CustomPolicy, env, verbose=0, learning_rate=1e-5, tensorboard_log="./ppo2fulltime105-3333333333/" )
#model = DQN(CustomPolicy_3, env, verbose=0, prioritized_replay=True, tensorboard_log="./ppo2full10-2/" )
#model = ACKTR(CustomPolicy, env, verbose=0, learning_rate=1e-6, tensorboard_log="./ppo2full10-2/" )
#model = PPO2.load("trader10year15m-32bk-reward-lr5-100m-2x256-20y-shaped-7", env, policy=CustomPolicy, tensorboard_log="./ppo2full10/" )
#model = PPO2.load("trader10year15m-32bk-reward-lr5-80m-2x256-20y-shaped-3", env, policy=CustomPolicy, tensorboard_log="./ppo2full10/" )
#model = PPO2.load("trader10year15m-32bk-lstm", env, policy=CustomPolicy_2, tensorboard_log="./ppo2full10-10/" )
#model.load("traderlstm.pkl")
#mode = PPO2.load('traderlstm2', env, verbose=0)
#model = PPO2(MlpLnLstmPolicy, env, verbose=1)
#model = ACKTR(MlpPolicy, env, nprocs=num_cpu)
#model.learn(total_timesteps=int(num_timesteps * 1.1), seed=seed)


def evaluate(model, num_steps=14400):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward
    """
    episode_rewards = [[0.0] for _ in range(env.num_envs)]
    obs = env.reset()
    for i in range(num_steps):
      # _states are only useful when using LSTM policies
      actions, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(actions)
      
      # Stats
      for i in range(env.num_envs):
          episode_rewards[i][-1] += rewards[i]
          if dones[i]:
              episode_rewards[i].append(0.0)

    mean_rewards =  [0.0 for _ in range(env.num_envs)]
    n_episodes = 0
    for i in range(env.num_envs):
        mean_rewards[i] = np.mean(episode_rewards[i])     
        n_episodes += len(episode_rewards[i])   

    # Compute mean reward
    mean_reward = np.mean(mean_rewards)
    print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

    return mean_reward


#mean_reward_before_train = evaluate(model, num_steps=14400)

n_timesteps = 10000000

# Multiprocessed RL Training
start_time = time.time()
#model.learn(total_timesteps=int(n_timesteps * 1.1), seed=0)
model.learn(n_timesteps)
total_time_multi = time.time() - start_time
model.save("sonic_ppo_10m")

model.learn(n_timesteps)
model.save("sonic_ppo_20m")
model.learn(n_timesteps)
model.save("sonic_ppo_30m")
model.learn(n_timesteps)
model.save("sonic_ppo_40m")
model.learn(n_timesteps)
model.save("sonic_ppo_50m")
model.learn(n_timesteps)
model.save("sonic_ppo_60m")
model.learn(n_timesteps)
model.save("sonic_ppo_70m")
model.learn(n_timesteps)
model.save("sonic_ppo_80m")
model.learn(n_timesteps)
model.save("sonic_ppo_90m")
model.learn(n_timesteps)
model.save("sonic_ppo_100m")
#model.save("trader10year15m-32bk-2m")
#model.save("trader10year15m-32bk-lstm")

print("Took {:.2f}s for multiprocessed version - {:.2f} FPS".format(total_time_multi, n_timesteps / total_time_multi))

mean_reward_before_train = evaluate(model, num_steps=14400)