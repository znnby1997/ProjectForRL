import gym
import ma_gym

from ma_gym_workspace.model.single_a2c_attention import A2CAttention
from ma_gym_workspace.model.maddpg import MADDPG
import matplotlib.pyplot as plt
import time
import numpy as np

train_episode_num = 10000
save_model_theshold = 100
agent_num = 5


def single_a2c_train():
    env = gym.make('Combat-v0')

    total_reward_list = []
    # local_reward_list = []

    model = A2CAttention(env.observation_space[0].shape[0], env.action_space[0].n, agent_num)

    for episode_ in range(train_episode_num):
        total_reward = 0
        # local_reward = 0
        obs_n = env.reset()
        # print('agent num', ma_gym.n_agents)
        # print('obs_n', obs_n)
        done_n = [False for _ in range(agent_num)]  # 这里每个agent的done值永远是一样的
        step_ = 0
        while not all(done_n):
            # print('obs num', len(obs_n))
            actions = model.infer_action(obs_n)
            # print(len(actions))

            next_obs_n, reward_n, done_n, info = env.step(actions)  # info里可以看到每个agent的health

            print('step info', info, 'local reward', reward_n)

            total_reward += sum(reward_n)

            for agent_i in range(agent_num):
                d_ = 0 if done_n[agent_i] else 1
                model.push_dynamics(obs_n[agent_i], actions[agent_i],
                                    reward_n[agent_i], next_obs_n[agent_i], d_)
            model.learn()

            step_ += 1
            obs_n = next_obs_n

        print('Episode %d is over \t Steps: %d \t total_reward: %0.2f' % (
            episode_, step_, total_reward
        ))
        total_reward_list.append(total_reward)
        # local_reward_list.append(local_reward)
    env.close()

    plt.plot(total_reward_list)
    # plt.plot(local_reward_list, color='b', label='local reward')
    # plt.legend(loc='best')
    plt.savefig('reward/a2c_attention_train_reward' + time.strftime('%y%m%d%H%M', time.localtime()) + '.png')
    plt.close()


def maddpg_train():
    env = gym.make('Combat-v0')

    total_reward_list = []
    model = MADDPG(env.observation_space[0].shape[0], agent_num, env.action_space[0].n)

    for episode_i in range(train_episode_num):
        total_reward = 0
        obs_n = env.reset()
        done_n = [False for _ in range(agent_num)]
        step_ = 0
        while not all(done_n):
            joint_actions = model.infer_actions(obs_n)
            print('joint_actions', joint_actions)
            next_obs_n, reward_n, done_n, info = env.step(joint_actions)
            print('step info:', info, 'local reward:', reward_n)

            total_reward += sum(reward_n)
            d_ = [0 if d else 1 for d in done_n]

            model.push_dynamics(obs_n, joint_actions, reward_n, next_obs_n, d_)
            model.learn()

            step_ += 1
            obs_n = next_obs_n
        print('Episode %d is over \t Step: %d \t total reward: %0.2f' % (episode_i, step_, total_reward))
        total_reward_list.append(total_reward)
    env.close()

    plt.plot(total_reward_list)
    plt.savefig('reward/maddpg_train_reward' + time.strftime('%y%m%d%H%M', time.localtime()) + '.png')
    plt.close()


if __name__ == '__main__':
    # single_a2c_train()
    maddpg_train()


