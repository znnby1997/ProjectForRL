import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random

import matplotlib.pyplot as plot

from collections import namedtuple
from torch.distributions import Categorical


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'is_end'])


class Net(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Net, self).__init__()

        self.h_layer = nn.Linear(state_dim, 20)
        self.h_layer.weight.data.normal_(0, 0.1)
        self.h_layer.bias.data.normal_(0, 0.1)

        self.out_layer = nn.Linear(20, action_dim)
        self.out_layer.weight.data.normal_(0, 0.1)
        self.out_layer.bias.data.normal_(0, 0.1)

    def forward(self, state):
        embedding = self.h_layer(state)
        embedding = f.relu(embedding)
        return self.out_layer(embedding)  # get Q(s_t) embedding, index is action


class ExpMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, is_end):
        if len(self.memory) < self.capacity:
            # insert a new tuple if and only if memory has enough capacity
            self.memory.append(None)

        self.memory[self.position] = Experience(state, action, reward, next_state, is_end)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, action_dim, state_dim,
                 batch_size=32, learning_rate=0.0001,
                 exp_pool_capacity=200, gamma=0.99, init_epsilon=1, final_epsilon=0.001,
                 entropy_weight=0.01):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.batch_size = batch_size
        self.exp_pool = ExpMemory(exp_pool_capacity)
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        # self.delta = (init_epsilon - final_epsilon) / 100000

        self.q_network = Net(action_dim, state_dim)  # type: Net

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def infer_action(self, state, step=1, greedy=False):
        # if self.epsilon > self.final_epsilon:
        if not greedy:
            delta = np.exp(-step) / 10000
            self.epsilon = max(self.final_epsilon, self.epsilon - delta)
        else:
            self.epsilon = 0

        state_tensor = torch.tensor(state, dtype=torch.float)

        if random.random() >= self.epsilon:
            with torch.no_grad():
                q_value = self.q_network.forward(state_tensor)  # q_s, indexed by action
                # print('q_s', q_value)
                action = q_value.max(0)[1].item()
                # print('action0:', action)
        else:
            action = random.randint(0, self.action_dim - 1)

        return action

    def learn(self):
        if len(self.exp_pool) < self.batch_size:
            print('Sample is not enough')
            return

        batch = self.exp_pool.sample(self.batch_size)
        data = Experience(*zip(*batch))

        state_batch = torch.tensor(data.state, dtype=torch.float)
        action_batch = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)
        reward_batch = torch.tensor(data.reward, dtype=torch.float)
        next_state_batch = torch.tensor(data.next_state, dtype=torch.float)
        is_end_batch = torch.tensor(data.is_end, dtype=torch.float)

        pi = self.q_network(state_batch)
        q_value = pi.gather(1, action_batch).squeeze(-1)  # shape: 32 * 1
        entropy = torch.mean(-Categorical(pi).entropy() * self.entropy_weight)
        # print(entropy)

        q_next = self.q_network(next_state_batch).detach()  # type: torch.Tensor
        # q_target = reward_batch + torch.mul(
        #     self.gamma * q_next.max(1)[0],
        #     is_end_batch
        # )
        q_target = reward_batch + self.gamma * q_next.max(1)[0] * is_end_batch

        # print('qvalue', q_value.shape)
        # print('qtarget', q_target.shape)

        loss = f.smooth_l1_loss(input=q_value, target=q_target) + entropy
        # loss = self.loss(target=q_target, input=q_value)
        self.optimizer.zero_grad()
        loss.mean().backward()
        # print('loss', loss.mean().item())
        self.optimizer.step()


episode_size = 10000
step_size = 300
test_size = 10
update_ratio = 10
save_theshold = 195


def test(en, episode_num=50, url='0.th'):
    m = DQN(en.action_space.n, en.observation_space.shape[0])
    m.q_network.load_state_dict(torch.load(url))
    scores = []

    for episode in range(episode_num):
        o = en.reset()
        score = 0

        for step in range(300):
            a = m.infer_action(o, greedy=True)
            s_, r, d, _ = en.step(a)

            score += r

            if d:
                print('Episode %d is over after %d steps \t total reward is %0.2f'
                      % (episode, step, score))
                break

            o = s_
        scores.append(score)

    plot.plot(scores)
    plot.show()
    plot.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = DQN(env.action_space.n, env.observation_space.shape[0])
    reward_list = []

    for episode_id in range(episode_size):
        obs = env.reset()
        total_reward = 0

        for step_ in range(step_size):
            # print('observation:', obs)
            a = model.infer_action(obs, step=step_)
            # print('action:', a)
            next_s, r, d, _ = env.step(a)

            total_reward += r

            d_ = 0 if d else 1

            # x, x_dot, theta, theta_dot = next_s
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            r = -1 if d else 0.1

            model.exp_pool.push(obs, a, r, next_s, d_)

            model.learn()

            if d:
                print('Episode %d is over after %d steps \t total reward is %0.2f'
                      % (episode_id, step_, total_reward))
                break

            obs = next_s
        reward_list.append(total_reward)

        if episode_id >= 200 and np.mean(reward_list[-200:]) >= 195:
            print('Successfully')
            torch.save(model.q_network.state_dict(), 'dqn_baseline_net_param.th')
            break

    plot.plot(reward_list)
    plot.show()
    plot.close()
    test(env)

