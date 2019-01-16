#from dumbrain.rl.retro_contest.install_games import installGamesFromDir
#colabInstallGames()
#installGamesFromDir(romdir='data/roms/')
#import retro
#list( filter( lambda game: game.startswith( 'Sonic' ), retro.list_games() ) )

import retrowrapper
import retro
#env = retrowrapper.RetroWrapper(
    #game='SonicTheHedgehog2-Genesis',
    #state='MetropolisZone.Act1', record='.'
#)



class RetroExample():
    def __init__(self):
        self.love = 'moose' 

    def train(self):
        env = retro.make(game='SonicTheHedgehog2-Genesis', state='MetropolisZone.Act1')
        obs = env.reset()
        while True:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                obs = env.reset()


#if __name__ == '__main__':
    #main()


#env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record='.')
#env.reset()
#while True:
    #_obs, _rew, done, _info = env.step(env.action_space.sample())
    #if done:
        #break