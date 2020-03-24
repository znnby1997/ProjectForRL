import gym
import ma_gym

env = gym.make('Switch2-v0')
print('agent num: ', ma_gym.n_agents)
done_n = [False for _ in range(ma_gym.n_agents)]
ep_reward = 0

obs_n = env.reset()
print('single agent obs', obs_n[0])

while not all(done_n):
    # env.render()
    print('action', type(env.action_space.sample()))
    obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
    print('reward: ', reward_n)
    ep_reward += sum(reward_n)
    print('ep_reward:', ep_reward)
env.close()

print('observation space: ', env.observation_space[0].shape[0])
print('action space: ', env.action_space[0].n)
