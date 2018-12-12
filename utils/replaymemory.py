import random


class ReplayMemoryDropStrategy:
    OLDEST = 'oldest'
    NEWEST = 'newest'
    RANDOM = 'random'


class ReplayMemory(list):
    def __init__(self, max_size, drop_strategy=ReplayMemoryDropStrategy.OLDEST):
        super().__init__()
        self.max_size = max_size
        self.drop_strategy = drop_strategy

    def append(self, item):
        if self.max_size != 0 and len(self) == self.max_size:
            if self.drop_strategy == ReplayMemoryDropStrategy.OLDEST:
                self.pop(0)
            elif self.drop_strategy == ReplayMemoryDropStrategy.NEWEST:
                self.pop()
            elif self.drop_strategy == ReplayMemoryDropStrategy.RANDOM:
                self.pop(random.randrange(len(self)))
            else:
                raise ValueError('Unknown DropStrategy')

        super().append(item)

    def __add__(self, ls):
        for item in ls:
            self.append(item)

    def __iadd__(self, ls):
        for item in ls:
            self.append(item)
