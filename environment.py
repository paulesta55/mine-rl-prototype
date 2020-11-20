import gym
import minerl
from gym import spaces
import numpy as np

treechop_env = gym.make("MineRLTreechop-v0")

class MyEnv(gym.Env):
    """Custom Neurocar Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = treechop_env.observation_space
        self.env_name = "MineRLTreechop-v0"

    def step(self, a_idx):
        # return treechop_env.step(actions_arr[action])
        a = treechop_env.action_space.noop()
        a["attack"] = 1
        if a_idx == 0:
            # forward
            a["forward"] = 1
            a["jump"] = 1
        elif a_idx == 1:
            # left
            a["camera"] = [0, -5]
        elif a_idx == 2:
            # right
            a["camera"] = [0, 5]
        elif a_idx == 3:
            # up
            a["camera"] = [-5, 0]
        elif a_idx == 4:
            # down
            a["camera"] = [5, 0]
        elif a_idx == 5:
            # forward
            a["forward"] = 1
        elif a_idx == 6:
            a["left"] = 1
        elif a_idx == 7:
            a["right"] = 1
        elif a_idx == 8:
            a["jump"] = 1
        elif a_idx == 9:
            a["back"] = 1
        return treechop_env.step(a)

    def reset(self):
        return treechop_env.reset()

    def render(self, mode='human'):
        return treechop_env.render(mode)

    def close (self):
        treechop_env.close()