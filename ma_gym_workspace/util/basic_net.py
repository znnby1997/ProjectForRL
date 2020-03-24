import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical


class ActorAttention(nn.Module):
    """
        每个agent的观测是一个150维的向量
    """
    def __init__(self, obs_dim, n_actions, hidden_dim=48):
        super(ActorAttention, self).__init__()
        self.fc_layer1 = nn.Linear(obs_dim, hidden_dim)

        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)

        self.fc_layer2 = nn.Linear(hidden_dim, n_actions)

    # pi
    def forward(self, x, softmax_dim=0):
        in_embedding = f.relu(self.fc_layer1(x))
        # attention_weights = f.softmax(self.attention_layer(in_embedding), dim=softmax_dim)
        # attention_embedding = in_embedding * attention_weights
        # action_prob = f.softmax(self.fc_layer2(attention_embedding), dim=softmax_dim)
        action_prob = f.softmax(self.fc_layer2(in_embedding), dim=softmax_dim)
        return action_prob


class CriticAttention(nn.Module):
    def __init__(self, obs_dim, hidden_dim=48, output_size=1):
        super(CriticAttention, self).__init__()
        self.fc_layer1 = nn.Linear(obs_dim, hidden_dim)

        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)

        self.fc_layer2 = nn.Linear(hidden_dim, output_size)

    # v
    def forward(self, obs, softmax_dim=0):
        in_embedding = f.relu(self.fc_layer1(obs))
        # attention_weights = f.softmax(self.attention_layer(in_embedding), dim=softmax_dim)
        # attention_embedding = in_embedding * attention_weights
        # value = self.fc_layer2(attention_embedding)
        value = self.fc_layer2(in_embedding)
        return value


class Actor(nn.Module):
    """MADDPG
    分部执行过程中的每个agent依据自己的观测单独执行动作
    MADDPG中直接输出一个动作,所以我们从网络中直接完成一个动作的采样(感觉应该无法计算梯度)
    """
    def __init__(self, single_obs_dim, n_single_action, hidden_dim=48):
        super(Actor, self).__init__()
        self.h_layer = nn.Linear(single_obs_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_single_action)

    # 离散动作使用gumbel softmax
    def forward(self, single_obs, temperature=1, gs_dim=0):
        embedding = f.relu(self.h_layer(single_obs))
        action_distribution = f.gumbel_softmax(self.out(embedding), temperature, dim=gs_dim)
        return action_distribution


class MACritic(nn.Module):
    """
    每个agent的Q都包含所有agent的观测以及每个agent执行的动作
    """
    def __init__(self, joint_obs_dim, n_agents, hidden_dim=48, output_size=1):
        super(MACritic, self).__init__()
        self.h_layer = nn.Linear(joint_obs_dim + n_agents, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_size)

    def forward(self, joint_obs, joint_actions, cat_dim=1):
        embedding = f.relu(self.h_layer(torch.cat((joint_obs, joint_actions), cat_dim)))
        return self.out(embedding)



