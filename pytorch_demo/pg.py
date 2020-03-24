import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


StepInfo = namedtuple('StepInfo', ['state', 'action', 'reward'])

# PG计算每步reward时，包含当前的即时reward以及之后的每步的reward


class PG(nn.Module):
    def __init__(self, state_dim, n_actions, gamma=0.99, entropy_weight=0.01,
                 lr=0.0001, hidden_dim=48):
        super(PG, self).__init__()

        self.h_layer = nn.Linear(state_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr)

        self.gamma = gamma
        self.entropy_weight = entropy_weight

        self.steps_list = []

    def forward(self):
        raise NotImplementedError

    def pi(self, state, softmax_dim=0):
        state_embedding = f.relu(self.h_layer(state))
        action_prob = f.softmax(self.fc_pi(state_embedding), dim=softmax_dim)
        return action_prob

    def store_step_info(self, s, a, r):
        self.steps_list.append(StepInfo(s, a, r))

    def infer_action(self, state, greedy=False):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_prob = self.pi(state)
            if not greedy:
                action_distributions = Categorical(action_prob)
                return action_distributions.sample().item()
            else:
                return action_prob.max(0)[1].item()

    # 计算累积的折扣rewrad
    def get_discount_norm_reward(self):
        # 创建一个每步总reward的存储空间
        discount_rewards = []

        # Gt = Rt+1 + gamma * Rt+2 + gamma^2 * Rt+3 + ... + gamma^(T-t-1) * RT
        for i in range(len(self.steps_list)):
            total_reward = 0
            j = i
            terminal_step = len(self.steps_list) - 1
            while j <= terminal_step:
                step_info = self.steps_list[j]  # type: StepInfo
                total_reward += (self.gamma**(j - i)) * step_info.reward
                j += 1
            discount_rewards.append(total_reward)

        # normalization
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards)
        return discount_rewards

    def learn(self):
        # 维度问题待解决
        rewards = torch.tensor(self.get_discount_norm_reward(), dtype=torch.float).reshape(-1, 1)
        data = StepInfo(*zip(*self.steps_list))

        states = torch.tensor(data.state, dtype=torch.float)
        actions = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)  # [batch, 1]

        # 定义损失函数
        pi = self.pi(states, softmax_dim=1)  # type: torch.Tensor
        actions_probs = pi.gather(1, actions)
        entropy = Categorical(pi).entropy().reshape(-1, 1)
        actor_loss = -torch.log(actions_probs) * rewards - self.entropy_weight * entropy

        # 梯度更新
        self.optimizer.zero_grad()
        actor_loss.mean().backward()
        self.optimizer.step()

        self.steps_list = []


episode_size = 30000
save_theshold = 198


def test(episode_num=1000, url='../model/pg_model.th'):
    env = gym.make('CartPole-v0')
    model = torch.load(url)
    scores = []
    for epi in range(episode_num):
        obs = env.reset()
        score = 0
        while True:
            # env.render()
            a = model.infer_action(obs, greedy=True)
            s_, r, d, _ = env.step(a)
            score += r

            if d:
                print('Episode %d is over | reward: %0.2f' % (epi, score))
                scores.append(score)
                break
            obs = s_

    plt.plot(scores)
    plt.show()
    plt.close()


def main():
    env = gym.make('CartPole-v0')
    model = PG(env.observation_space.shape[0], env.action_space.n)  # type: PG
    episode_reward_list = []

    for episode in range(episode_size):
        obs = env.reset()

        episode_reward = 0

        while True:
            action = model.infer_action(obs)

            obs_, reward, done, _ = env.step(action)

            episode_reward += reward
            reward = -1 if done else 0.1

            model.store_step_info(obs, action, reward)

            if done:
                print('Episode %d is over | reward %0.2f' % (episode, episode_reward))
                episode_reward_list.append(episode_reward)
                model.learn()
                break

            obs = obs_

        if episode >= 200 and np.mean(episode_reward_list[-200:]) >= save_theshold:
            print('Successfully')
            torch.save(model, 'pg_model0.th')
            break

    plt.plot(episode_reward_list)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # main()
    test()









