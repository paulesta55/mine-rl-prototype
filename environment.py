import gym
import minerl
from gym import spaces
import numpy as np

treechop_env = gym.make("MineRLTreechop-v0")
actions_arr = np.repeat(treechop_env.action_space.noop(), 6)
# forward
actions_arr[0]["forward"] = 1
# left
actions_arr[1]["camera"] = [0, -1]
# right
actions_arr[2]["camera"] = [0, 1]
# up
actions_arr[3]["camera"] = [-1, 0]
# down
actions_arr[4]["camera"] = [1, 0]
# attack
actions_arr[5]["attack"] = 1

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