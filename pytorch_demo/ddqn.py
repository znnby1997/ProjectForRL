import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random

import matplotlib.pyplot as plot

from collections import namedtuple


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'is_end'])


class Net(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(Net, self).__init__()

        self.h_layer = nn.Linear(state_dim, 48)
        self.h_layer.weight.data.normal_(0, 0.1)
        # self.h_layer.bias.data.normal_(0, 0.1)

        self.out_layer = nn.Linear(48, action_dim)
        self.out_layer.weight.data.normal_(0, 0.1)
        # self.out_layer.bias.data.normal_(0, 0.1)

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


class DDQN(object):
    def __init__(self, action_dim, state_dim,
                 batch_size=32, learning_rate=0.001,
                 exp_pool_capacity=10000, gamma=0.99, init_epsilon=0.4, final_epsilon=0.01):
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.batch_size = batch_size
        self.exp_pool = ExpMemory(exp_pool_capacity)
        self.gamma = gamma
        self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.delta = (init_epsilon - final_epsilon) / 200

        self.cur_q_network = Net(action_dim, state_dim)  # type: Net
        self.tar_q_network = Net(action_dim, state_dim)  # type: Net
        self.tar_q_network.load_state_dict(self.cur_q_network.state_dict())  # type: Net

        self.optimizer = optim.Adam(self.cur_q_network.parameters(), lr=learning_rate)

    def infer_action(self, state):
        # if self.epsilon > self.final_epsilon:
        self.epsilon -= self.delta

        state_tensor = torch.tensor(state, dtype=torch.float)

        if random.random() >= self.epsilon:
            with torch.no_grad():
                q_value = self.cur_q_network.forward(state_tensor)  # q_s, indexed by action
                action = torch.argmax(q_value).item()
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
        reward_batch = torch.tensor(data.reward, dtype=torch.float).reshape(-1, 1)
        next_state_batch = torch.tensor(data.next_state, dtype=torch.float)
        is_end_batch = torch.tensor(data.is_end, dtype=torch.float).reshape(-1, 1)

        q_value = self.cur_q_network(state_batch).gather(1, action_batch)  # shape: 32 * 1

        cur_best_actions = self.cur_q_network(next_state_batch).detach().max(1)[1].reshape(-1, 1)
        q_next = self.tar_q_network(next_state_batch).detach()  # type: torch.Tensor
        q_target = reward_batch + torch.mul(
            self.gamma * q_next.gather(1, cur_best_actions),
            is_end_batch
        )
        loss = f.smooth_l1_loss(input=q_value, target=q_target)
        # loss = self.loss(target=q_target, input=q_value)
        # print('loss', loss.mean())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def update_tar_net(self):
        self.tar_q_network.load_state_dict(self.cur_q_network.state_dict())


episode_size = 3000
step_size = 300
test_size = 10
update_ratio = 10


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = DDQN(env.action_space.n, env.observation_space.shape[0])
    reward_list = []

    for episode_id in range(episode_size):
        obs = env.reset()
        total_reward = 0

        for step_ in range(step_size):
            if step_ % update_ratio == 0:
                model.update_tar_net()

            a = model.infer_action(obs)
            next_s, r, d, _ = env.step(a)

            total_reward += r
            model.exp_pool.push(obs, a, r, next_s, d)

            model.learn()

            if d:
                print('Episode %d is over after %d steps \t total reward is %0.2f'
                      % (episode_id, step_, total_reward))
                break

            obs = next_s
        reward_list.append(total_reward)

    plot.plot(reward_list)
    plot.show()

