# third-party imports
import numpy as np


class CustomLinearRegression:
    def __init__(self):
        self._fitted = False

    def _init_params(self, X):
        # no of training examples, no of features
        self._m, self._n = X.shape

        # weight initialization
        self._W = np.zeros(self._n)
        self._b = 0

    def fit(self, X, y, epochs=1000, lr=0.05):

        self._init_params(X)

        # gradient descent learning
        for _ in range(epochs):

            Y_pred = X.dot(self._W) + self._b

            # calculate gradients
            dW = - (2 * X.T.dot(y - Y_pred)) / self._m
            db = - 2 * np.sum(y - Y_pred) / self._m

            # update weights
            self._W = self._W - lr * dW
            self._b = self._b - lr * db

        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            raise Exception('Please call `SimpleLinearRegression.fit(X, y)` before making predictions.')

        return X.dot(self._W) + self._b
