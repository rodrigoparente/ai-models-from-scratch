# third-party imports
import numpy as np


def one_hot_encode(y):
    number_classes = len(set(y))
    return np.eye(number_classes)[y.reshape(-1)]


def standardized(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)
