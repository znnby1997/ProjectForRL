import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as f
import torch.optim as optim

from ma_gym_workspace.util.basic_net import Actor, MACritic
from ma_gym_workspace.util.experience_memory import ExperienceMemory, Dynamics


class MADDPG(object):
    def __init__(self, obs_dim, n_agents, n_actions, capacity=5000, batch_size=300,
                 gamma=0.99, a_lr=0.0001, c_lr=0.0001, tau=0.02, lt=0.0001, ut=1.0, step_weight=0.9):
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lt = lt
        self.ut = ut
        self.step = 0
        self.step_weight = step_weight
        self.temperature = 1.0
        self.single_obs_dim = obs_dim
        self.single_n_actions = n_actions

        self.memory = ExperienceMemory(capacity)

        # cur 用于训练，tar 用于采样
        self.actors_cur = []  # type: list[Actor]
        self.critics_cur = []  # type: list[MACritic]
        self.actors_tar = []   # type: list[Actor]
        self.critics_tar = []  # type: list[MACritic]
        self.actors_optimizer = []  # type: list[optim.Adam]
        self.critics_optimizer = []  # type: list[optim.Adam]

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.init_agents()

    def init_agents(self):
        for agent_i in range(self.n_agents):
            self.actors_cur.append(Actor(self.single_obs_dim, self.single_n_actions))
            self.critics_cur.append(MACritic(self.single_obs_dim * self.n_agents, self.n_agents))
            self.actors_tar.append(Actor(self.single_obs_dim, self.single_n_actions))
            self.critics_tar.append(MACritic(self.single_obs_dim * self.n_agents, self.n_agents))
            self.actors_optimizer.append(optim.Adam(self.actors_cur[agent_i].parameters(), lr=self.a_lr))
            self.critics_optimizer.append(optim.Adam(self.critics_cur[agent_i].parameters(), lr=self.c_lr))

        # 对tar网络采取软更新策略
        self.actors_tar = self.update_tar(self.actors_cur, self.actors_tar, self.tau)
        self.critics_tar = self.update_tar(self.critics_cur, self.critics_tar, self.tau)

    def infer_actions(self, joint_obs, add_noise=True):
        actions = []
        if add_noise:
            self.temperature = self.update_temperature(self.ut, self.lt, self.step_weight)
            print('temperature:', self.temperature)
            for agent_i, actor_tar in enumerate(self.actors_tar):
                state = torch.tensor(joint_obs[agent_i], dtype=torch.float)
                action_dis = actor_tar.forward(state, temperature=self.temperature).detach()  # type: torch.Tensor
                # temperature应该随着训练进行从1逐渐变小，要慢慢减少噪声
                actions.append(action_dis.max(0)[1].item())
        else:
            for agent_i, actor_tar in enumerate(self.actors_tar):
                state = torch.tensor(joint_obs[agent_i], dtype=torch.float)
                action_dis = actor_tar.forward(state, temperature=0).detach()  # type: torch.Tensor
                # temperature应该随着训练进行从1逐渐变小，要慢慢减少噪声
                actions.append(action_dis.max(0)[1].item())
        return actions

    def update_temperature(self, init_t, final_t, step_weight):
        temperature = pow(1.0 - (final_t / init_t), (self.step / step_weight))
        self.step += 1
        return max(temperature, final_t)

    def push_dynamics(self, joint_obs, joint_actions, n_rewards, next_joint_obs, n_done):
        self.memory.push(joint_obs, joint_actions, n_rewards, next_joint_obs, n_done)

    def learn(self):
        """
        先计算Critic loss，包含了每个agent的观测和动作作为输入
        之后计算actor loss，仅包含每个agent自己的观测
        :return:
        """
        if len(self.memory) < self.batch_size:
            print('Data is not enough')
            return

        batch = self.memory.sample(self.batch_size)
        data = Dynamics(*zip(*batch))
        joint_obs = torch.tensor(data.state, dtype=torch.float)
        joint_actions = torch.tensor(data.action, dtype=torch.float)
        n_rewards = torch.tensor(data.reward, dtype=torch.float)
        joint_next_obs = torch.tensor(data.next_state, dtype=torch.float)
        n_dones = torch.tensor(data.is_end, dtype=torch.float)

        for agent_i, (actor_tar, actor_cur, critic_tar, critic_cur, actor_o, critic_o) in \
            enumerate(zip(self.actors_tar, self.actors_cur,
                          self.critics_tar, self.critics_cur, self.actors_optimizer, self.critics_optimizer)):
            """
            计算Critic loss:
            target value: y = ri + gamma * Qi 注意，这里的Q采用target网络计算，动作也由target网络给出
            current value: Qi 这里的Q采用current网络计算，动作来自经验回放池采样，梯度更新的也是cur参数
            """
            # print('single agent obs', joint_next_obs[:, agent_i:agent_i+1, :].squeeze(),
            #        'shape', joint_next_obs[:, agent_i: agent_i+1, :].squeeze().shape)
            # print('action', Categorical(self.actors_tar[0].forward(joint_next_obs[:, 0:1, :].squeeze(), softmax=1)).sample().shape)
            next_actions = torch.cat([tar.forward(joint_next_obs[:, idx:idx+1, :].squeeze(), gs_dim=1).max(1)[1].reshape(-1, 1) for idx, tar in enumerate(self.actors_tar)], dim=1).float()
            # print('next_actions', next_actions.shape)
            # print('single rewards', n_rewards[:, agent_i:agent_i+1].shape,
            #     'single done', n_dones[:, agent_i:agent_i+1].shape)
            # print('joint next obs shape', joint_next_obs.shape, 'joint next action shape', next_actions.shape)
            target_qs = n_rewards[:, agent_i:agent_i+1] + self.gamma * critic_tar(joint_next_obs.reshape(self.batch_size, -1), next_actions) * n_dones[:, agent_i:agent_i+1]
            # print('joint obs shape', joint_obs.reshape(300, -1).shape, 'joint actions', joint_actions.shape)
            current_qs = critic_cur(joint_obs.reshape(self.batch_size, -1), joint_actions)
            critic_loss = f.smooth_l1_loss(current_qs, target_qs.detach())
            critic_o.zero_grad()
            critic_loss.mean().backward()
            critic_o.step()

            """
            计算Actor loss:
            目测好像目标函数就是Q值，aj是通过actor cur获得的，其他的a是经验回放中获得的
            """
            # print('joint obs shape', joint_obs[:, agent_i:agent_i+1, :].squeeze().shape)
            action_i = actor_cur.forward(joint_obs[:, agent_i:agent_i+1, :].squeeze(), 1).max(1)[1].reshape(-1, 1)
            # print('joint action', action_i.shape)
            joint_actions[:, agent_i:agent_i+1] = action_i
            actor_loss = - critic_cur(joint_obs.reshape(self.batch_size, -1), joint_actions)
            actor_o.zero_grad()
            actor_loss.mean().backward()
            actor_o.step()

        self.actors_tar = self.update_tar(self.actors_cur, self.actors_tar, self.tau)
        self.critics_tar = self.update_tar(self.critics_cur, self.critics_tar, self.tau)

    @staticmethod
    def update_tar(agents_cur, agents_tar, tau):
        """
        软更新的方式，更新步长为tau
        :param agents_cur:
        :param agents_tar:
        :param tau:
        :return:
        """
        for agent_cur, agent_tar in zip(agents_cur, agents_tar):
            key_list = list(agent_cur.state_dict().keys())
            state_dict_t = agent_tar.state_dict()
            state_dict_c = agent_cur.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key] * tau + state_dict_t[key] * (1 - tau)
            agent_tar.load_state_dict(state_dict_t)
        return agents_tar
