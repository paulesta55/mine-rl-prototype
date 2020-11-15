import gym
import minerl
from gym import spaces
import numpy as np

treechop_env = gym.make("MineRLTreechop-v0")

values = np.linspace(-180,180, 16)
actions_arr = np.repeat(treechop_env.action_space.noop(), 35)

for i in range(16):
    actions_arr[i]["camera"] = [0,values[i]]

for i in range(16,32):
    actions_arr[i]["camera"] = [0, values[i-16]]

actions_arr[32]["forward"] = 1
actions_arr[33]["forward"] = 1
actions_arr[33]["jump"] = 1
actions_arr[34]["attack"] = 1

class MyEnv(gym.Env):
    """Custom Neurocar Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MyEnv, self).__init__()
        self.actions_arr = actions_arr
        self.action_space = spaces.Discrete(actions_arr.shape[0])
        self.observation_space = treechop_env.observation_space

    def step(self, action):
        return treechop_env.step(actions_arr[action])

    def reset(self):
        return treechop_env.reset()

    def render(self, mode='human'):
        return treechop_env.render(mode)

    def close (self):
        treechop_env.close()