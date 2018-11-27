from prettytable import PrettyTable
from agents.agent import Agent
import numpy as np
import random


class QLearningAgent(Agent):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space.n, action_space.n))

        self.learning_rate = 0.1
        self.discount_factor = 0.99

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.001
        self.exploration_rate_min = 0.01

    def remember(self, state, action, reward, next_state, done):
        self.q_table[state, action] = \
            (1 - self.learning_rate) * self.q_table[state, action] \
            + self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]))

    def act(self, state, prod=False):
        if prod or random.uniform(0, 1) > self.exploration_rate:
            # Follow policy
            action = np.argmax(self.q_table[state])
        else:
            # Explore
            action = self.action_space.sample()

        return action

    def before(self):
        pass

    def after(self):
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate = self.exploration_rate * (1 - self.exploration_rate_decay)
        else:
            self.exploration_rate = self.exploration_rate_min

    def print_q_table(self):
        # columns = range(self.action_space.n)

        columns = ['0 (Left)', '1 (Down)', '2 (Right)', '3 (Up)']
        columns = list(map(lambda c: 'Action {}'.format(c), columns))
        columns.insert(0, '')

        table = PrettyTable(columns)

        for i, state in enumerate(self.q_table):
            state = state.tolist()
            state.insert(0, 'State {}'.format(i + 1))

            table.add_row(state)

        print(table)
