from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def before(self):
        pass

    @abstractmethod
    def after(self):
        pass
