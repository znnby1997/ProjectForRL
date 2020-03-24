import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from torch.distributions import Categorical
from ma_gym_workspace.util.basic_net import ActorAttention, CriticAttention
from ma_gym_workspace.util.experience_memory import ExperienceMemory, Dynamics
import numpy as np


class A2CAttention(object):
    """
    打算为每个agent创建一个ac，每个agent的ac并不用于训练
    额外创建一个cur ac，用于训练，每个agent的ac从cur中传递参数
    经实测，a_lr在attention下要设置为0.00001
    """
    def __init__(self, state_dim, n_actions, n_agents, gamma=0.99, batch_size=128,
                 capacity=5000, entropy_weight=0.01, a_lr=0.0001, c_lr=0.001):
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.exp_pool = ExperienceMemory(capacity)
        self.n_agents = n_agents

        self.actor_cur = ActorAttention(state_dim, n_actions)  # type: ActorAttention
        self.critic_cur = CriticAttention(state_dim, n_actions)  # type: CriticAttention
        self.actor_optimizer = optim.Adam(self.actor_cur.parameters(), lr=a_lr)
        self.critic_optimizer = optim.Adam(self.critic_cur.parameters(), lr=c_lr)

        self.actors_tar = []  # type: list[ActorAttention]
        self.critics_tar = []  # type: list[CriticAttention]

        for agent_i in range(self.n_agents):
            self.actors_tar.append(ActorAttention(state_dim, n_actions))
            self.critics_tar.append(CriticAttention(state_dim, n_actions))
            self.actors_tar[agent_i].load_state_dict(self.actor_cur.state_dict())
            self.critics_tar[agent_i].load_state_dict(self.critic_cur.state_dict())

    def infer_action(self, joint_states, greedy=False):
        """
        每个agent的动作都是一个数值，5个agent合起来构成了一个list
        :param joint_states:
        :param greedy:
        :return:
        """
        # # print('state', state)
        # state = torch.tensor(state, dtype=torch.float)
        #
        # action_dis = self.actor_net.forward(state).detach()  # type: torch.Tensor
        # # print('action_distribution', action_dis.numpy())
        # with torch.no_grad():
        #     if not greedy:
        #         return Categorical(action_dis).sample().item()
        #     else:
        #         return action_dis.max(0)[1].item()
        # print('agent num', len(self.actors_tar))
        actions = []
        for agent_i, actor in enumerate(self.actors_tar):
            state = torch.tensor(joint_states[agent_i], dtype=torch.float)
            action_dis = actor.forward(state).detach()  # type: torch.Tensor
            if not greedy:
                actions.append(Categorical(action_dis).sample().item())
            else:
                actions.append(action_dis.max(0)[1].item())
        return actions

    def push_dynamics(self, state, action, reward, next_state, done):
        self.exp_pool.push(state, action, reward, next_state, done)

    def update_tar(self):
        for actor_tar, critic_tar in zip(self.actors_tar, self.critics_tar):
            actor_tar.load_state_dict(self.actor_cur.state_dict())
            critic_tar.load_state_dict(self.critic_cur.state_dict())

    def learn(self):
        if len(self.exp_pool) <= self.batch_size:
            print('Data is not enough.')
            return

        batch = self.exp_pool.sample(self.batch_size)
        data = Dynamics(*zip(*batch))

        states = torch.tensor(data.state, dtype=torch.float)
        actions = torch.tensor(data.action, dtype=torch.long).reshape(-1, 1)
        rewards = torch.tensor(data.reward, dtype=torch.float)
        next_states = torch.tensor(data.next_state, dtype=torch.float)
        is_ends = torch.tensor(data.is_end, dtype=torch.float)
        # print('reward', rewards.shape)

        # define actor loss
        td_target = rewards + self.gamma * self.critic_cur.forward(next_states, 1).squeeze(-1) * is_ends
        # print('td_target', td_target.shape)
        advantages = td_target - self.critic_cur.forward(states, 1).squeeze(-1)
        pi = self.actor_cur.forward(states, 1)  # type: torch.Tensor
        prob_a = pi.gather(1, actions).squeeze(-1)
        # print('prob_a', prob_a.shape)
        entropy = Categorical(pi).entropy()
        actor_loss = - torch.log(prob_a) * advantages.detach() - entropy * self.entropy_weight

        # update actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        for name, param in self.actor_cur.named_parameters():
            print('name:', name, 'param gradient:', param.grad)
        self.actor_optimizer.step()

        # define critic loss
        critic_loss = f.smooth_l1_loss(self.critic_cur.forward(states, 1).squeeze(-1), td_target.detach())

        # update critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.mean().backward()
        self.critic_optimizer.step()

        self.update_tar()
