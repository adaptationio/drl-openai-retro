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

class PPO2_SB():
    def __init__(self, num_env=16):
        self.love = 'Ramona'
        
        self.env = SubprocVecEnv([lambda: make_env(stack=False, scale_rew=True)] * num_env)
        self.model = PPO2(policy=CnnPolicy,
                      env=self.env,
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


    def train(self, n_timesteps, save):
        self.model.learn(n_timesteps)
        self.model.save(save)
    
    
    def evaluate(self, num_steps=14400):
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
            actions, _states = self.model.predict(obs)
            # # here, action, rewards and dones are arrays
            # # because we are using vectorized env
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
        