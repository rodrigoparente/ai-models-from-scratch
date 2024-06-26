# third-party imports
import numpy as np


def train_test_split(X, y, test_size=0.25):

    X_copy = np.copy(X)
    y_copy = np.copy(y)

    n_rols = len(X_copy)
    test_size = int(n_rols * test_size)

    X_train = X_copy[test_size + 1: n_rols]
    y_train = y_copy[test_size + 1: n_rols]

    X_test = X_copy[0: test_size]
    y_test = y_copy[0: test_size]

    return X_train, X_test, y_train, y_test
