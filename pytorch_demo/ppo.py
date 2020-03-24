import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical

from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

StepInfo = namedtuple('stepinfo', ['state', 'action', 'reward'])


class ACModel(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=48):
        super(ACModel, self).__init__()
        self.h_layer = nn.Linear(state_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_actions)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def forward(self):
        raise NotImplementedError

    def pi(self, state, softmax_dim=0):
        state_ = f.relu(self.h_layer(state))
        return f.softmax(self.fc_pi(state_), softmax_dim)

    def v(self, state):
        state_ = f.relu(self.h_layer(state))
        return self.fc_v(state_)


class PPO(object):
    def __init__(self, state_dim, n_actions, epsilon=0.2, gamma=0.99, entropy_weight=0.01,
                 lr=0.0001):
        self.epsilon = epsilon
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.step_list = []

        # old ac用于采样，ac用于训练
        self.old_ac = ACModel(state_dim, n_actions)  # type: ACModel
        self.ac = ACModel(state_dim, n_actions)  # type: ACModel
        self.optimizer = optim.Adam(self.ac.parameters(), lr)
        self.old_ac.load_state_dict(self.ac.state_dict())

    def store_step_info(self, s, a, r):
        self.step_list.append(StepInfo(s, a, r))

    def infer_action(self, state, greedy=False):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            action_prob = self.old_ac.pi(state)
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
        for i in range(len(self.step_list)):
            total_reward = 0
            j = i
            terminal_step = len(self.step_list) - 1
            while j <= terminal_step:
                step_info = self.step_list[j]  # type: StepInfo
                total_reward += (self.gamma**(j - i)) * step_info.reward
                j += 1
            discount_rewards.append(total_reward)

        # normalization
        discount_rewards -= np.mean(discount_rewards)
        discount_rewards /= np.std(discount_rewards)
        return discount_rewards

    def learn(self):
        rewards = torch.tensor(self.get_discount_norm_reward(), dtype=torch.float).reshape(-1, 1)
        data = StepInfo(*zip(*self.step_list))

        states = torch.tensor(data.state, dtype=torch.float)
        actions = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)  # [batch, 1]

        # 定义损失函数
        pi_old = self.old_ac.pi(states, softmax_dim=1)  # type: torch.Tensor
        actions_old_prob = pi_old.gather(1, actions)
        pi = self.ac.pi(states, softmax_dim=1)  # type: torch.Tensor
        actions_prob = pi.gather(1, actions)

        entropy = Categorical(pi).entropy().reshape(-1, 1)

        # 这一步的具体意义还没有理解
        ratios = torch.exp(torch.log(actions_prob) - torch.log(actions_old_prob.detach()))  # 这种方式训练较快
        # ratios = actions_prob / actions_old_prob.detach() 这种方式训练较慢

        states_value = self.ac.v(states)

        advantages = rewards - states_value.detach()

        policy1 = ratios * advantages
        policy2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

        loss = - torch.min(policy1, policy2) + f.smooth_l1_loss(states_value, rewards) - self.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.step_list = []

        self.old_ac.load_state_dict(self.ac.state_dict())


episode_size = 10000
save_theshold = 195


def test(episode_num=1000, url='../model/ppo_model.th'):
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
    env = gym.make('MountainCar-v0')
    model = PPO(env.observation_space.shape[0], env.action_space.n)  # type: PG
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

        # if episode >= 200 and np.mean(episode_reward_list[-200:]) >= save_theshold:
        #     print('Successfully')
        #     torch.save(model, 'ppo_model.th')
        #     break

    plt.plot(episode_reward_list)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    # test()

