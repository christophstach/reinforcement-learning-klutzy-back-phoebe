import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import FullyConnectedDeepQLearningAgent
from utils import moving_average

env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = 1000

agent = FullyConnectedDeepQLearningAgent(
    env_name,
    env.observation_space,
    env.action_space,
    exploration_rate_decay=0.0001,
    memory_size=0
)

episodes = 10000
simulate = True

timesteps_history = []
total_reward_history = []

for episode in tqdm(range(episodes)):
    total_reward = 0
    timestep = 0
    state = env.reset()

    agent.before()
    while True:
        timestep += 1

        if simulate and episode >= episodes - 5:
            env.render()
            time.sleep(0.02)

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward = total_reward + reward

        if done:
            timesteps_history.append(timestep)
            total_reward_history.append(total_reward)

            break

    agent.after()

# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(moving_average(total_reward_history, 500))
# ax2.plot(moving_average(timesteps_history, 500))

plt.plot(moving_average(total_reward_history, 100))
plt.show()
