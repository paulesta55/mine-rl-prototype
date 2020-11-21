from network import ReplayMemory, DQN, Transition
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import logging
import minerl

logging.basicConfig(level=logging.DEBUG)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = minerl.data.make(env.env_name)
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = env.action_space.n
img_height = 64
img_width = 64

policy_net = DQN(img_height, img_width, n_actions).to(device)
target_net = DQN(img_height, img_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0

def converter(observation):
    region_size = 8
    obs = observation['pov']
    obs = obs / 25
    H,W,C = obs.shape
    state = torch.from_numpy(obs).float().to(device)
    if len(state.shape) < 4:
            state = torch.unsqueeze(state, 0)
    state = state.reshape((-1,C,H,W))
    return state

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def pretraining_step():
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    optimizer.zero_grad()
    pred_action = policy_net(state_batch)
    loss = F.cross_entropy(pred_action, action_batch)
    logging.debug(f"loss = {loss}")
    loss.backward()
    optimizer.step()
    return loss.item()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

traj_names = data.get_trajectory_names()
np.random.shuffle(traj_names)
for n in traj_names[:50]:
    for state, action, reward, next_state, done in data.load_data(n, skip_interval=4):
        camera_threshold = (abs(action['camera'][0]) + abs(action['camera'][1])) / 2.0
        if camera_threshold > 2.5:
            # pitch +5
            if ((action['camera'][0] > 0) & (
                    abs(action['camera'][0]) > abs(action['camera'][1]))):
                action_idx = 0
            # pitch -5
            elif ((action['camera'][0] < 0) & (
                    abs(action['camera'][0]) > abs(action['camera'][1]))):
                action_idx = 1
            # yaw +5
            elif ((action['camera'][1] > 0) & (
                    abs(action['camera'][0]) < abs(action['camera'][1]))):
                action_idx = 2
            # yax -5
            elif ((action['camera'][1] < 0) & (
                    abs(action['camera'][0]) < abs(action['camera'][1]))):
                action_idx = 3
        # forward
        elif action["forward"] == 1:
            action_idx = 4
            # forward and jump
            if action["jump"] == 1:
                action_idx = 5
        # left
        elif action["left"] == 1:
            action_idx = 6
        # right
        elif action["right"] == 1:
            action_idx = 7
        # back
        elif action["back"] == 1:
            action_idx = 8
        # jump
        else:
            action_idx = 9
        memory.push(state, action_idx, next_state, reward)

num_pretraining = 10000

loss_history = []
for i in range(num_pretraining):
    loss_history.append(pretraining_step())

np.save('loss_history',np.array(loss_history))

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)

policy_net.apply(weights_init)

from environment import MyEnv

env = MyEnv()

num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = converter(env.reset())
    avg_rew = 0;
    for t in count():
        # Select and perform an action
        action = select_action(state)
        obs, rew, _, _ = env.step(action.item())
        reward = torch.tensor([rew], device=device)
        next_state = converter(obs)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        logging.debug(f"current reward = {rew}")
        avg_rew += rew
    avg_rew /= t
    logging.info(f"avg reward = {avg_rew}")
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
