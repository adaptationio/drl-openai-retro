#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
#import gym_remote.exceptions as gre
from ..agents.sonic_util import AllowBacktracking, make_env
#from sonic_util import AllowBacktracking, make_env
import retrowrapper
import retro
import csv
import os


#from sonic_util import AllowBacktracking, make_env
#env = retro.make(game='SonicTheHedgehog2-Genesis', state='MetropolisZone.Act1')
class Rainbow():
    def __init__(self):
        self.love = 'Ramona'
        self.env_fns = [] 
        self.env_names = []
        self.environs = ['SpringYardZone.Act3', 'SpringYardZone.Act2', 'GreenHillZone.Act3','GreenHillZone.Act1','StarLightZone.Act2','StarLightZone.Act1','MarbleZone.Act2','MarbleZone.Act1','MarbleZone.Act3','ScrapBrainZone.Act2','LabyrinthZone.Act2','LabyrinthZone.Act1','LabyrinthZone.Act3', 'SpringYardZone.Act1','GreenHillZone.Act2','StarLightZone.Act3','ScrapBrainZone.Act1']

    def create_envs(self, n):
            for i in self.environs:            
                self.env_fns.append(partial(make_env, game='SonicTheHedgehog-Genesis', state=i, stack=False, scale_rew=False))
                self.env_names.append('SonicTheHedgehog-Genesis' + '-' + i)

    def train(self):
        """Run DQN until the environment throws an exception."""
        self.env_fns, self.env_names = self.create_envs() 
        self.env = BatchedFrameStack(batched_gym_env(env_fns), num_images=4, concat=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True # pylint: disable=E1101
        with tf.Session(config=config) as sess:
            dqn = DQN(*rainbow_models(sess,
                                    self.env.action_space.n,
                                    gym_space_vectorizer(env.observation_space),
                                    min_val=-421,
                                    max_val=421))
            player = NStepPlayer(BatchedPlayer(self.env, dqn.online_net), 3)
            optimize = dqn.optimize(learning_rate=1e-4)
            sess.run(tf.global_variables_initializer())
            
            dqn.train(num_steps=100000000, # Make sure an exception arrives before we stop.
                    player=player,
                    replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                    optimize_op=optimize,
                    train_interval=1,
                    target_interval=64,
                    batch_size=32,
                    min_buffer_size=25000,
                    handle_ep=_handle_ep,
                    num_envs = len(self.env_fns),
                    save_interval=10)

        
    
                  

   

