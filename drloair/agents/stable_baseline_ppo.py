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
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
#from stable_baselines.deepq.policies import FeedForwardPolicy
#from template_env import Template_Gym

#env = Template_Gym()
from ..agents.retro_util import AllowBacktracking, make_env

import retrowrapper
import retro

timestamp = datetime.datetime.now().strftime('%y%m%d%H%M%S')

class PPO2_SB():
    def __init__(self):
        self.love = 'Ramona'
        self.env_fns = [] 
        self.env_names = []
        self.environs = ['SpringYardZone.Act3', 'SpringYardZone.Act2', 'GreenHillZone.Act3','GreenHillZone.Act1','StarLightZone.Act2','StarLightZone.Act1','MarbleZone.Act2','MarbleZone.Act1','MarbleZone.Act3','ScrapBrainZone.Act2','LabyrinthZone.Act2','LabyrinthZone.Act1','LabyrinthZone.Act3', 'SpringYardZone.Act1','GreenHillZone.Act2','StarLightZone.Act3','ScrapBrainZone.Act1']
        self.environsv2 = ['1Player.Axel.Level1']
        self.generate_expert_traj = generate_expert_traj
    
    def create_envs(self, game_name, state_name, num_env):
        
        for i in range(num_env):            
            self.env_fns.append(partial(make_env, game=game_name, state=state_name))
            self.env_names.append(game_name + '-' + state_name)
        self.env = SubprocVecEnv(self.env_fns)
    

    def train(self, game, state, num_e=1, n_timesteps=25000000, save='default2'):
        self.create_envs(game_name=game, state_name=state, num_env=num_e)
        #self.model = PPO2.load("default2", SubprocVecEnv(self.env_fns), policy=CnnPolicy, tensorboard_log="./sonic/" )
        #self.model = PPO2(CnnPolicy, SubprocVecEnv(self.env_fns), learning_rate=1e-5, verbose=1,tensorboard_log="./sonic/" )

        self.model = PPO2(policy=CnnPolicy,
                      env=SubprocVecEnv(self.env_fns),
                      n_steps=8192,
                      nminibatches=8,
                      lam=0.95,
                      gamma=0.99,
                      noptepochs=4,
                      ent_coef=0.001,
                      learning_rate=lambda _: 2e-5,
                      cliprange=lambda _: 0.2,
                      verbose=1,
                      tensorboard_log="./sonic/")
        self.model.learn(n_timesteps)
        self.model.save(save)
        self.model.learn(n_timesteps)
        self.model.save(save+'2')
        self.model.learn(n_timesteps)
        self.model.save(save+'3')
        self.model.learn(n_timesteps)
        self.model.save(save+'4')
    
    
    def evaluate(self, game, state, num_e=1, num_steps=14400):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward
        """
        
        self.create_envs(game_name=game, state_name=state, num_env=num_e)
        self.model = PPO2.load("default2", SubprocVecEnv(self.env_fns), policy=CnnPolicy, tensorboard_log="./sonic/" )
        episode_rewards = [[0.0] for _ in range(self.env.num_envs)]
        obs = self.env.reset()
        for i in range(num_steps):
            # _states are only useful when using LSTM policies
            actions, _states = self.model.predict(obs)
            # # here, action, rewards and dones are arrays
            # # because we are using vectorized env
            obs, rewards, dones, info = self.env.step(actions)
      
      # Stats
            for i in range(self.env.num_envs):
                episode_rewards[i][-1] += rewards[i]
                if dones[i]:
                    episode_rewards[i].append(0.0)

        mean_rewards =  [0.0 for _ in range(self.env.num_envs)]
        n_episodes = 0
        for i in range(self.env.num_envs):
            mean_rewards[i] = np.mean(episode_rewards[i])     
            n_episodes += len(episode_rewards[i])   

    # Compute mean reward
        mean_reward = np.mean(mean_rewards)
        print("Mean reward:", mean_reward, "Num episodes:", n_episodes)

        return mean_reward

    def pre_train(self):
        # Using only one expert trajectory
        # you can specify `traj_limitation=-1` for using the whole dataset
        dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                                                                                                traj_limitation=1, batch_size=128)

        model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
        # Pretrain the PPO2 model
        model.pretrain(dataset, n_epochs=1000)

        # As an option, you can train the RL agent
        # model.learn(int(1e5))

        # Test the pre-trained model
        env = model.get_env()
        obs = env.reset()

        reward_sum = 0.0
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

        env.close()

    def gen_pre_train(self, game, state, num_e=1, save='default2', episodes=10):
        self.create_envs(game_name=game, state_name=state, num_env=num_e)
        env=SubprocVecEnv(self.env_fns)
        self.expert_agent = "moose"
        self.generate_expert_traj(self.expert_agent, save, env, n_episodes=episodes)
        

