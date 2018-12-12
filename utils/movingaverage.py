import numpy as np


def movingaverage(values, window):
    if len(values) >= window:
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')
    else:
        return [0]
