import argparse

import gym
import numpy as np
import argparse
from drloair import *
parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="choose algorithm, eg PPO, rainbow, ",
                    type=str)
parser.add_argument("--envt", help="Select Retro environment",
                    type=str)
args = parser.parse_args()
if args.algo == 'rainbow' or 'Rainbow':
    agent = Rainbow()
elif args.algo == 'ppo' or 'PPO':
    agent =



args = parser.parse_args()
print(args.square**2)

def main():
    pass
    
    

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    




