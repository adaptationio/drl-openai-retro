"""
Environments and wrappers for Sonic training.
"""

import gym
import numpy as np
import retro

from baselines.common.atari_wrappers import WarpFrame, FrameStack
#import gym_remote.client as grc

def make_env(game=None, state=None, stack=True, scale_rew=True, allowbacktrace=False, custom=False):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock')
    #env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
    #env = retro.make(game='StreetsOfRage2-Genesis', state='1Player.Axel.Level1')
    #env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile')
    #env = retro.make(game='SuperMarioWorld-Snes', state='Bridges1')
    env = retro.make(game=game, state=state)
    #SuperMarioWorld-Snes ['Bridges1',

    env = SonicDiscretizerV2(env)
    #env = StreetOfRage2Discretizer(env)
    #env = StreeFighter2Discretizer(env)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if custom:
        env = CustomGym(env)
    if allowbacktrace:
        env = AllowBacktracking(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SonicDiscretizerV2(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizerV2, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SonicDiscretizerV3(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizerV3, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['RIGHT'], ['RIGHT', 'DOWN'], ['DOWN','B'], ['RIGHT','B'], []]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class StreetOfRage2Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(StreetOfRage2Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B'], [], ['LEFT', 'B'], ['RIGHT', 'B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class StreeFighter2Discretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(StreeFighter2Discretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'],
                   ['DOWN', 'B'], ['B'], ['DOWN', 'A'], ['A'], ['DOWN', 'C'], ['C']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class SuperMarioWorldDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SuperMarioWorldDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['RIGHT', 'B'], ['B'], ['RIGHT', 'A'], ['A'],
                   ['RIGHT', 'C'], ['C'], ['RIGHT', 'Y'], ['Y'], ['RIGHT', 'X'], ['X'], ['RIGHT', 'Z'], ['Z']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.

    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self.env.render()
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class CustomGym(gym.Wrapper):
    """
    add custom features
    """
    def __init__(self, env):
        super(CustomGym, self).__init__(env)

    def reset(self, **kwargs): # pylint: disable=E0202
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        self.env.render()
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info