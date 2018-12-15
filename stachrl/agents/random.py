from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def remember(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return self.action_space.sample()

    def before(self):
        pass

    def after(self):
        pass
