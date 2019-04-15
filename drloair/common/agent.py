import numpy as np
from keyboard import KeyLogger
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

class HumanAgent():
    def __init__(self):
        self.train = True
        self.keyboard_logger = KeyLogger()

    def train(self, _obs):
        self.action = self.keyboard_logger.actions_step()
        return self.action


    


    


    
