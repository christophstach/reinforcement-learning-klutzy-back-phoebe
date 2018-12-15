import os
import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from stachrl.agents import DQNAgent
from stachrl.utils import movingaverage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

episodes = 1
timesteps = 2000
simulate = True

env_name = 'CartPole-v0'
env = gym.make(env_name)
env._max_episode_steps = timesteps

agent = DQNAgent(
    name=env_name,
    state_shape=env.observation_space.shape,
    action_size=env.action_space.n,
    exploration_rate_decay=0.005,
    memory_size=2500
)

total_reward_history = []
iterator = tqdm(range(episodes))

for episode in iterator:
    iterator.set_postfix_str('Reward: {:.2f}, ε: {:.2f}'.format(
        movingaverage(total_reward_history, 50)[-1],
        agent.exploration_rate
    ))

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
            total_reward_history.append(total_reward)
            break

    agent.after()

f, (ax1) = plt.subplots(1, 1)
ax1.plot(movingaverage(total_reward_history, 500))

plt.show()
