import argparse

import gym
import numpy as np
import argparse
from dumbrain.rl.retro_contest.install_games import installGamesFromDir
installGamesFromDir(romdir='data/roms/')
from drloair import *
parser = argparse.ArgumentParser()
parser.add_argument("--algo", help="choose algorithm, eg PPO, rainbow, ",
                    type=str)
parser.add_argument("--game", help="Select Retro environment",
                    type=str)
parser.add_argument("--state", help="Select Retro environment State",
                    type=str)
args = parser.parse_args()
if args.algo == 'rainbow' or 'Rainbow':
    agent = Rainbow()
elif args.algo == 'ppo' or 'PPO':
    agent = PPO2_SB()
else:
    agent = PPO2_SB()
args = parser.parse_args()
if args.game:
    game_arg = args.game
else:
    game_arg = 'SonicTheHedgehog-Genesis'
if args.state:
    state_arg = args.state
else:
    state_arg = 'GreenHillZone.Act1'
    




def main():
    agent = PPO2_SB()
    agent.train(game=game_arg, state=state_arg)
    #agent.evaluate(game=game_arg, state=state_arg)
    agent.gen_pre_train(game=game_arg, state=state_arg)
    
    

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
    




