import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from stachrl.agents import QLearningAgent
from stachrl.utils import movingaverage, clearscreen

episodes = 30000
timesteps = 500
simulate = True

env_name = 'FrozenLake-v0'
env = gym.make(env_name)
env._max_episode_steps = timesteps

agent = QLearningAgent(
    name=env_name,
    state_space=env.observation_space,
    action_space=env.action_space
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
            clearscreen()
            print('############## Episode: {}, {} ##############\n\n'.format(episode + 1, timestep + 1))
            env.render()
            time.sleep(0.25)

        action = agent.act(state)

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        total_reward = total_reward + reward

        if done:
            timesteps_history.append(timestep)
            total_reward_history.append(total_reward)

            if simulate and episode >= episodes - 1:
                if reward:
                    print('*** YOU WIN ***')
                else:
                    print('*** YOU LOOSE ***')
                    pass

            break

    agent.after()

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(movingaverage(total_reward_history, 500))
ax2.plot(movingaverage(timesteps_history, 500))

plt.show()
