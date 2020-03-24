import gym
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import numpy as np
from collections import namedtuple
import random

import matplotlib.pyplot as plot

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'is_end'])

######################################################################
# 返回确定的动作，而非动作的概率分布
# 四个网络：
#   Actor: current network 通过梯度更新
#          target network 通过从current network参数复制获得
#   Critic: current network 通过梯度更新
#           target network 通过从current network参数复制获得
######################################################################


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=20, output_size=2):
        super(Actor, self).__init__()
        self.h_layer = nn.Linear(state_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)

    # pi
    def forward(self, state):
        h_layer = self.h_layer(state)
        h_layer = f.relu(h_layer)
        action = self.out(h_layer)
        action = torch.tanh(action)
        # print("state, h_layer, action")
        # print(state, h_layer, action)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=20, output_size=1):
        super(Critic, self).__init__()
        self.h_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)

    # q
    def forward(self, state, action):
        h_layer = self.h_layer(torch.cat((state, action), dim=1))
        h_layer = f.relu(h_layer)
        q_value = self.out(h_layer)
        # print("state, action, h_layer, q_value")
        # print(state, action, h_layer, q_value)
        return q_value


class ExpMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, is_end):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(state, action, reward, next_state, is_end)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 01   32batch  0.01lr   tanh相加
# 02                       
# 03   32hidden  64ba
#  04  64    64   相对最好
# 05   64    128
# 06   32    32

# 07   64  64   0.001
#  08             0.005
# 09   07的sigma   0.075  0.1
# 10   2000
#  11   reward/10,  0.05  0.025
#  12                            tau0.05  3w  0.005  0.001
# 13   和之前一样的
# #  14 ou后期为0
# 15    /100
# 16     /500



version = 21920

class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, tau=0.05,
                 gamma=0.99, init_epsilon=1.0, final_epsilon=0.01, hidden_dim=64,
                 batch_size=64, pool_size=10000, actor_lr=0.005, critic_lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.exp_pool = ExpMemory(pool_size)
        self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.action_bound = action_bound
        self.tau = tau

        # four networks
        self.target_actor = Actor(state_dim, hidden_dim)  # type: Actor
        self.current_actor = Actor(state_dim, hidden_dim)  # type: Actor
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)  # type: Critic
        self.current_critic = Critic(state_dim, action_dim, hidden_dim)  # type: Critic
        self.target_actor.load_state_dict(self.current_actor.state_dict())
        self.target_critic.load_state_dict(self.current_critic.state_dict())

        self.actor_optimizer = optim.Adam(self.current_actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.current_critic.parameters(), lr=critic_lr)

    @staticmethod
    def ou(a_v, mu=0, theta=0.05, sigma=0.025):
        """Ornstein-Uhlenbeck process
        formula: ou = theta * (mu - a_v) + sigma * w
        公式中的w表征布朗运动，即一种外界随机噪声

        :param a_v: 动作
        :param mu: 均值
        :param theta: 离均值的改变幅度
        :param sigma: 随机噪声的权重
        :return: OU value
        """
        # if episode > 500:
            # return -0.01
        return theta * (mu - a_v) + sigma * np.random.rand(1)

    def infer_action(self, state, episode, step=1, greedy=False):
        # 动作为连续动作，使用clip可以划定范围
        state_embedding = torch.tensor(state, dtype=torch.float)
        action_ = self.current_actor(state_embedding).detach()
        action__ = torch.sum(action_).numpy()
        # print(action__)

        if not greedy:
            # need to add a noise
            # delta = np.exp(-step) / 10000
            # self.epsilon = max(self.final_epsilon, self.epsilon - delta)
            # noise = self.epsilon * self.ou(action_)
            action = np.clip(action__ + self.ou(action__), -self.action_bound + 0.01, self.action_bound - 0.01)
            print("action--", state, action_, action__, action)
            return action
        else:
            return action_

    def net_update(self, target_net: nn.Module, net: nn.Module):
        # 很有用的网络参数软更新方法
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            source = target_param.data * (1.0 - self.tau) + param.data * self.tau
            index_len = param.shape[0]
            index = torch.arange(start=0, end=index_len, step=1, dtype=torch.long)
            target_param.data.index_copy_(dim=0, index=index, source=source)

    def learn(self):
        if len(self.exp_pool) < self.batch_size:
            print('Sample is not enough')
            return

        batch = self.exp_pool.sample(self.batch_size)
        data = Experience(*zip(*batch))

        state_batch = torch.tensor(data.state, dtype=torch.float)
        action_batch = torch.tensor(data.action, dtype=torch.float)
        reward_batch = torch.tensor(data.reward, dtype=torch.float).reshape(-1, 1)
        next_state_batch = torch.tensor(data.next_state, dtype=torch.float)
        is_end_batch = torch.tensor(data.is_end, dtype=torch.float).reshape(-1, 1)

        # update current actor network
        actions = self.current_actor(state_batch).sum(1).unsqueeze(1)
        # print(actions, state_batch)
        q_values = self.current_critic(state_batch, actions)
        cur_actor_loss = - q_values  # type: torch.Tensor
        # print('current actor network loss: ', cur_actor_loss.mean().data.numpy())
        self.actor_optimizer.zero_grad()
        cur_actor_loss.mean().backward()
        self.actor_optimizer.step()

        # update current critic network
        next_actions = self.target_actor(next_state_batch).detach().sum(1).unsqueeze(1)
        td_target = reward_batch + self.gamma * self.target_critic(next_state_batch,
                                                                   next_actions).detach()
        cur_critic_loss = f.smooth_l1_loss(input=self.current_critic(state_batch, action_batch), target=td_target.detach())
        # print('current critic network loss: ', cur_critic_loss.mean().data.numpy())
        self.critic_optimizer.zero_grad()
        cur_critic_loss.mean().backward()
        self.critic_optimizer.step()

        # self.net_update(self.target_actor, self.current_actor)
        # self.net_update(self.target_critic, self.current_critic)


def test(en, episode_num=50, url='../model/ddpg_model.th'):
    m = torch.load(url)
    scores = []

    for episode in range(episode_num):
        o = en.reset()
        score = 0
        for step in range(300):
            a = m.infer_action(o, episode, greedy=True)
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


episode_size = 1000
step_size = 300
test_size = 30
param_update_rate = 10
save_theshold = -5


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    model = DDPG(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high[0])
    total_reward_list = []
    for episode in range(episode_size):
        obs = env.reset()
        total_reward = 0
        for step_ in range(step_size):
            if step_ % param_update_rate == 0:
                model.net_update(model.target_actor, model.current_actor)
                model.net_update(model.target_critic, model.current_critic)

            a = model.infer_action(obs, episode, step=step_)
            n_s, r, d, _ = env.step(a)

            print(r)
            total_reward += r

            r /= 10  # 放缩尺度

            d_ = 0 if d else 1
            model.exp_pool.push(obs, a, r, n_s, d_)

            model.learn()

            if d:
                total_reward_list.append(total_reward)
                print('The episode %d is over after %d steps | reward: %0.2f' % (episode, step_, total_reward), np.mean(total_reward_list[-50:]))
                break

            obs = n_s

        if episode >= 200 and np.mean(total_reward_list[-50:]) >= save_theshold:
            print('Successfully')
            torch.save(model, '../model/{}.th'.format(version))
            # break

    plot.plot(total_reward_list)
    # plot.savefig(f"logs/{version}.png")
    plot.show()
    plot.close()






