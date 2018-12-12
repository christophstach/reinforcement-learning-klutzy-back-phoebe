import os
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from agents.agent import Agent
from utils import ReplayMemory


class DDQNAgent(Agent):
    def __init__(
            self,
            name,
            state_shape,
            action_size,
            learning_rate=0.01,
            discount_factor=0.95,
            exploration_rate=1,
            exploration_rate_decay=0.005,
            exploration_rate_min=0.01,
            memory_size=10000,
            replay_sample_size=32,
            production=False,
            auto_save=True,
            auto_load=True
    ):
        self.production = production
        self.auto_save = auto_save
        self.auto_load = auto_load
        self._name = name
        self._model_path = 'models/{}-DDQN.h5'.format(self._name)
        self._state_shape = state_shape
        self._action_size = action_size

        self._learning_rate = learning_rate  # alpha
        self._discount_factor = discount_factor  # gamma

        self.exploration_rate = exploration_rate  # epsilon
        self._exploration_rate_decay = exploration_rate_decay
        self._exploration_rate_min = exploration_rate_min

        self._memory = ReplayMemory(memory_size)
        self._replay_sample_size = replay_sample_size

        self.network = self._build_model(self._model_path)

    def _build_model(self, filepath):
        model = Sequential([
            Dense(24, input_shape=self._state_shape),
            Activation('relu'),

            Dense(24),
            Activation('relu'),

            Dense(self._action_size),
            Activation('linear')
        ])

        model.compile(loss='mse', optimizer=Adam(lr=self._learning_rate))

        if self.auto_load and os.path.isfile(filepath):
            model.load_weights(filepath)

        return model

    def remember(self, state, action, reward, next_state, done):
        if not self.production:
            self._memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.production or random.uniform(0, 1) > self.exploration_rate:
            # Follow policy
            action = np.argmax(self.network.predict(state[np.newaxis]))
        else:
            # Explore
            action = random.randrange(self._action_size)

        return action

    def replay(self, batch_size):
        if len(self._memory) >= batch_size:
            batch = random.sample(self._memory, batch_size)
            targets = []
            states = []

            for state, action, reward, next_state, done in batch:
                target = self.network.predict(state[np.newaxis]).squeeze()

                if done:
                    target[action] = reward
                else:
                    target[action] = \
                        reward + \
                        self._discount_factor * \
                        np.max(self.network.predict(next_state[np.newaxis]))

                states.append(state)
                targets.append(target)

            self.network.fit(
                np.array(states),
                np.array(targets),
                verbose=0,
                epochs=1,
                callbacks=[
                    ModelCheckpoint(filepath=self._model_path) if self.auto_save and not self.production else None
                ]
            )

    def before(self):
        pass

    def after(self):
        if not self.production:
            self.replay(self._replay_sample_size)

            self.exploration_rate = max(
                self._exploration_rate_min,
                self.exploration_rate * (1 - self._exploration_rate_decay)
            )
