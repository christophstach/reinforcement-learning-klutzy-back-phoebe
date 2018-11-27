import random

import numpy as np
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.models import Sequential

from agents.agent import Agent
from utils import MaxSizeList


class ConvolutionDeepQLearningAgent(Agent):
    def __init__(self, name, state_space, action_space):
        self.name = name
        self.model_path = 'models/{}.h5'.format(self.name)
        self.state_space = state_space
        self.action_space = action_space

        self.learning_rate = 0.1
        self.discount_factor = 0.99

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.001
        self.exploration_rate_min = 0.01

        self.memory = MaxSizeList(10000)

        self.dqn = Sequential([
            Conv2D(16, kernel_size=3, input_shape=(self.state_space.shape[0], self.state_space.shape[1], 1)),
            Activation('relu'),

            Conv2D(16, kernel_size=3),
            Activation('relu'),

            Conv2D(16, kernel_size=3),
            Activation('relu'),

            Flatten(),

            Dense(24),
            Activation('relu'),

            Dense(24),
            Activation('relu'),

            Dense(self.action_space.n),
            Activation('linear')
        ])

        self.dqn.compile(loss='mse', optimizer='adam')

    def remember(self, state, action, reward, next_state, done):
        state = self.pre_process(state)
        next_state = self.pre_process(next_state)

        self.memory.append((state, action, reward, next_state, done))

    def pre_process(self, state):
        # Greyscale
        state = np.dot(state[..., :3], [0.299, 0.587, 0.114])
        state = np.expand_dims(state, axis=-1)

        return state

    def act(self, state):
        if random.uniform(0, 1) > self.exploration_rate:
            # Follow policy
            state = self.pre_process(state)
            action = np.argmax(self.dqn.predict(state[np.newaxis]))
        else:
            # Explore
            action = self.action_space.sample()

        return action

    def replay(self, batch_size):
        if len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)
            targets = []
            states = []

            for state, action, reward, next_state, done in batch:
                target = self.dqn.predict(state[np.newaxis]).squeeze()

                if done:
                    target[action] = reward
                else:
                    target[action] = \
                        (1 - self.learning_rate) * target[action] \
                        + self.learning_rate * (
                                reward + self.discount_factor * np.max(self.dqn.predict(next_state[np.newaxis]))
                        )

                states.append(state)
                targets.append(target)

            self.dqn.fit(
                np.array(states),
                np.array(targets),
                verbose=0
            )

    def before(self):
        pass

    def after(self):
        self.replay(32)

        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate = self.exploration_rate * (1 - self.exploration_rate_decay)
        else:
            self.exploration_rate = self.exploration_rate_min
