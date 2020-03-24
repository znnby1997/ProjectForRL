import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from torch.distributions import Categorical
from collections import namedtuple

import numpy as np
import random
import matplotlib.pyplot as plot


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'is_end'])


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=20):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    # pi function
    def forward(self, state, softmax_dim=0):
        h_layer = self.fc(state)
        h_layer = f.relu(h_layer)
        action_embedding = self.out(h_layer)

        # normalization
        # softmax such that the sum of actions is 1
        action_prob = f.softmax(action_embedding, dim=softmax_dim)
        return action_prob


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=20, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)

    # v function
    def forward(self, state):
        out = self.fc(state)
        out = f.relu(out)
        out = self.out(out)
        return out


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


class A2C(object):
    def __init__(self, state_dim, action_dim, capacity=200,
                 gamma=0.99, entropy_weight=0.01,
                 lr_actor=0.0001, lr_critic=0.0001, batch_size=32):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = ActorNetwork(state_dim, action_dim)  # type: ActorNetwork
        self.critic = CriticNetwork(state_dim)  # type: CriticNetwork

        self.exp_pool = ExpMemory(capacity)

        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def inter_action(self, state, greedy=False):
        state = torch.tensor(state, dtype=torch.float)
        action_prob = self.actor(state)
        if not greedy:
            action_prob = action_prob.data.numpy()
            # print(action)
            # action space: {0, 1}
            theshold = action_prob[0]
            if np.random.random() > theshold:
                return 1
            else:
                return 0
        else:
            action = action_prob.max(0)[1].item()
            return action

    def learn(self):
        if len(self.exp_pool) < self.batch_size:
            print('Sample is not enough')
            return

        batch = self.exp_pool.sample(self.batch_size)
        data = Experience(*zip(*batch))

        states = torch.tensor(data.state, dtype=torch.float)
        actions = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)
        rewards = torch.tensor(data.reward, dtype=torch.float)
        next_states = torch.tensor(data.next_state, dtype=torch.float)
        is_ends = torch.tensor(data.is_end, dtype=torch.float)

        td_target = rewards + self.gamma * self.critic(next_states).squeeze(-1) * is_ends  # type: torch.Tensor

        advantages = td_target - self.critic(states).squeeze(-1)  # type: torch.Tensor
        # cross_entropy for softmax function
        pi = self.actor.forward(states, softmax_dim=1)
        # print(pi.data.numpy().shape)
        # print(actions.data.numpy().shape)
        prob_a = pi.gather(1, actions).squeeze(-1)
        entropy = Categorical(pi).entropy()

        # update actor network parameters
        actor_loss = -torch.log(prob_a) * advantages.detach() - self.entropy_weight * entropy

        # print('policy loss: ', -torch.log(prob_a) * advantages.detach(),
        #       '\t cross-entropy loss: ', self.entropy_weight * entropy)
        # print('policy loss: ', actor_loss.mean())

        self.optimizer_actor.zero_grad()
        actor_loss.mean().backward()
        self.optimizer_actor.step()

        # update critic network parameters
        critic_loss = f.smooth_l1_loss(self.critic(states).squeeze(-1), td_target.detach())

        # print('critic loss: ', f.smooth_l1_loss(self.critic.forward(states), td_target.detach()))
        print('critic loss: ', critic_loss)

        self.optimizer_critic.zero_grad()
        critic_loss.mean().backward()
        self.optimizer_critic.step()


def test(en, episode_num=50, url='../model/a2c_model.th'):
    m = torch.load(url)
    scores = []

    for episode in range(episode_num):
        o = en.reset()
        score = 0

        for step in range(300):
            a = m.inter_action(o, greedy=True)
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


episode_size = 10000
step_size = 300
test_size = 10
update_ratio = 20
save_theshold = 195


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = A2C(env.observation_space.shape[0], env.action_space.n)
    reward_list = []

    for episode_id in range(episode_size):
        obs = env.reset()
        total_reward = 0

        for step_ in range(step_size):

            a = model.inter_action(obs)
            next_s, r, d, _ = env.step(a)

            total_reward += r

            r = -1 if d else 0.1
            d_ = 0 if d else 1
            model.exp_pool.push(obs, a, r, next_s, d_)

            model.learn()

            if d:
                print('Episode %d is over after %d steps \t total reward is %0.2f'
                      % (episode_id, step_, total_reward))
                break

            obs = next_s
        reward_list.append(total_reward)

        if episode_id >= 200 and np.mean(reward_list[-200:]) >= save_theshold:
            print('Successfully')
            torch.save(model, 'a2c_model.th')
            break

    plot.plot(reward_list)
    plot.show()
    plot.close()
    # test(env)








