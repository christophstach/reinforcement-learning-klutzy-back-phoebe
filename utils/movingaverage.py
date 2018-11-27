import numpy as np


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, 'valid')
