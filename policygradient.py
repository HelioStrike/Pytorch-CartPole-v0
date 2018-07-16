import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
lr = 1e-2
gamma = 0.99
batch_size = 5

running_gamma = 1
class PG(nn.Module):
    def __init__(self):
        super(PG, self).__init__()
        self.l1 = nn.Linear(4, 24)
        self.l2 = nn.Linear(24, 36)
        self.l3 = nn.Linear(36, 1)

    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return F.sigmoid(out)

def plotProgress(arr):
    plt.figure(1)
    plt.plot(arr)
    plt.pause(0.001)

reward_progress = []

policy = PG()
optimizer = optim.RMSprop(policy.parameters(), lr=lr)

state_pool = []
action_pool = []
reward_pool = []

for e in count():
    state = env.reset()

    for i in count(1):
        state = torch.from_numpy(state).float()
        probs = policy(state)
        m = Bernoulli(probs)
        action = m.sample()
        action = action.data.numpy().astype(int)[0]

        next_state, reward, done, _ = env.step(action)

        if done:
            reward = 0

        state_pool.append(state)
        action_pool.append(float(action))
        reward_pool.append(reward)

        state = next_state

        if done:
            print("Reward: ", i)
            reward_progress.append(i)
            plotProgress(reward_progress)
            break

    if e > 0 and e % batch_size == 0:
        running_add = 0

        for i in reversed(range(len(state_pool))):
            if(reward_pool[i] == 0):
                running_add = 0
            else :
                running_add = running_add*gamma + reward_pool[i]
                reward_pool[i] = running_add

        reward_pool  = np.array(reward_pool)
        reward_pool = (reward_pool - reward_pool.mean())/reward_pool.std()

        optimizer.zero_grad()
        for j in range(len(state_pool)):
            state = state_pool[j]
            action = torch.tensor(action_pool[j]).float()
            reward = np.int(reward_pool[j])

            probs = policy(state)
            m = Bernoulli(probs)
            loss = -reward*m.log_prob(action)*running_gamma

            loss.backward()
        optimizer.step()
        running_gamma *= gamma

        state_pool = []
        action_pool = []
        reward_pool = []

#DT20184497897
