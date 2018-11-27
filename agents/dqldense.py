import os
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model

from agents.agent import Agent
from utils import MaxSizeList


class DQLDense(Agent):
    def __init__(
            self,
            name,
            state_space,
            action_space,
            learning_rate=0.1,
            discount_factor=0.99,
            exploration_rate=1,
            exploration_rate_decay=0.001,
            exploration_rate_min=0.01,
            memory_size=10000,
            replay_sample_size=32
    ):
        self.production = False
        self.name = name
        self.model_path = 'models/{}-DQLDense.h5'.format(self.name)
        self.state_space = state_space
        self.action_space = action_space

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.exploration_rate = exploration_rate
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min

        self.memory = MaxSizeList(memory_size)
        self.replay_sample_size = replay_sample_size

        if os.path.isfile(self.model_path):
            self.dqn = load_model(self.model_path)
        else:
            self.dqn = Sequential([
                Dense(24, input_shape=self.state_space.shape),
                Activation('relu'),

                Dense(24),
                Activation('relu'),

                Dense(self.action_space.n),
                Activation('linear')
            ])

            self.dqn.compile(loss='mse', optimizer='adam')

    def remember(self, state, action, reward, next_state, done):
        if not self.production:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.production or random.uniform(0, 1) > self.exploration_rate:
            # Follow policy
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
                verbose=0,
                callbacks=[
                    ModelCheckpoint(filepath=self.model_path)
                ]
            )

    def before(self):
        pass

    def after(self):
        if not self.production:
            self.replay(self.replay_sample_size)

            self.exploration_rate = max(
                self.exploration_rate_min,
                self.exploration_rate * (1 - self.exploration_rate_decay)
            )
