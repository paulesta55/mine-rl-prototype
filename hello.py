import gym
import minerl

env = gym.make("MineRLTreechop-v0")

obs = env.reset()
done = False

while not done:
    act = env.action_space.noop()
    act["back"] = 0
    act["forward"] = 1
    act["jump"] = 1
    act["attack"] = 1
    obs, rew, done, _ = env.step(act)
    # env.render()