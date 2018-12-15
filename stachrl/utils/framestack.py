import numpy as np
from collections import deque


class FrameStack:
    def __init__(self, size, frame_shape):
        self._size = size
        self._frame_shape = frame_shape
        self._stack = deque(maxlen=self._size)
        self.clear()

    def append(self, frame):
        self._stack.append(frame)

    def get(self):
        return np.stack(self._stack, axis=-1)

    def clear(self):
        for _ in range(self._size):
            self._stack.append(np.zeros(self._frame_shape))
