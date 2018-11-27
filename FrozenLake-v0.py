import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import QLearningAgent
from utils import moving_average, clear_screen

env = gym.make('FrozenLake-v0')
agent = QLearningAgent(env.observation_space, env.action_space)

episodes = 9000
timesteps = 100

timesteps_history = []
total_reward_history = []

for episode in tqdm(range(episodes)):
    total_reward = 0
    state = env.reset()

    agent.before()
    for timestep in range(timesteps):
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

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(moving_average(total_reward_history, 500))
# ax2.plot(moving_average(timesteps_history, 500))

plt.show()

agent.print_q_table()

if False:
    for episode in range(5):
        state = env.reset()

        agent.before()
        for timestep in range(timesteps):
            action = agent.act(state, prod=True)
            clear_screen()
            print('############## Episode: {}, {} ##############\n\n'.format(episode + 1, timestep + 1))
            env.render()
            time.sleep(0.1)

            next_state, reward, done, info = env.step(action)
            #agent.remember(state, action, reward, next_state)

            if done:
                # clear_screen()
                # print('############## Episode: {}, {} ##############\n\n'.format(episode + 1, timestep + 1))
                # env.render()
                if reward:
                    print('*** YOU WIN ***')
                else:
                    print('*** YOU LOOSE ***')
                    pass

                # time.sleep(5)
                # clear_screen()
                break

        agent.after()
