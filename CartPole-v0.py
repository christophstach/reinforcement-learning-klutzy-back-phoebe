import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import DQLDense
from utils import moving_average

episodes = 1
timesteps = 100
simulate = True

env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = timesteps

agent = DQLDense(
    name=env_name,
    state_space=env.observation_space,
    action_space=env.action_space,
    exploration_rate_decay=0.001,
    memory_size=2500
)

timesteps_history = []
total_reward_history = []

for episode in tqdm(range(episodes)):
    total_reward = 0
    state = env.reset()

    agent.before()
    for timestep in range(timesteps):
        if simulate and episode >= episodes - 1:
            agent.production = True
            env.render()
            time.sleep(0.02)

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        reward = reward - abs(next_state[0])

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward = total_reward + reward

        if done:
            timesteps_history.append(timestep)
            total_reward_history.append(total_reward)

            break
    agent.after()

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(moving_average(total_reward_history, 500))
ax2.plot(moving_average(timesteps_history, 500))

plt.show()
