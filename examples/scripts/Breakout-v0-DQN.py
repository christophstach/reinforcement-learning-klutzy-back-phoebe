import os
import time

import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from stachrl.agents import DQNAgent
from stachrl.utils import movingaverage

from keras.layers import Dense, Activation, Conv2D, Flatten, LSTM, TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

episodes = 500
timesteps = 2000
simulate = True

env_name = 'Breakout-v0'
env = gym.make(env_name)
env._max_episode_steps = timesteps


class BreakoutAgent(DQNAgent):
    def _build_model(self, filepath):
        model = Sequential([
            Conv2D(filters=16, kernel_size=3, input_shape=self._state_shape),
            Activation('relu'),

            Conv2D(filters=16, kernel_size=3),
            Activation('relu'),

            Conv2D(filters=16, kernel_size=3),
            Activation('relu'),

            TimeDistributed(Flatten()),

            LSTM(24),

            Dense(24),
            Activation('relu'),

            Dense(self._action_size),
            Activation('linear')
        ])

        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))

        if self.auto_load and os.path.isfile(filepath):
            model.load_weights(filepath)

        return model


agent = BreakoutAgent(
    name=env_name,
    state_shape=env.observation_space.shape,
    action_size=env.action_space.n,
    exploration_rate_decay=0.005,
    memory_size=2500,
    auto_save=False,
    auto_load=False
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
