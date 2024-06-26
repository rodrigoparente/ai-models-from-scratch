# third-party imports
import numpy as np


class CustomLogisticRegression:
    def __init__(self):
        self._fitted = False

    def _init_params(self, X):
        # no of training examples, no of features
        self._m, self._n = X.shape

        # weight initialization
        self._weights = np.zeros(self._n)
        self._bias = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, epochs=1000, learning_rate=0.05):
        self._init_params(X)

        # gradient descent learning
        for _ in range(epochs):
            z = X.dot(self._weights) + self._bias
            y_pred = self._sigmoid(z)

            # calculate gradients
            dW = np.dot(X.T, (y_pred - y.T).reshape(self._m)) / self._m
            db = np.sum((y_pred - y.T).reshape(self._m)) / self._m

            # update weights
            self._weights = self._weights - learning_rate * dW
            self._bias = self._bias - learning_rate * db

        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            raise Exception('Please call `SimpleLogisticRegression.fit(X, y)` before making predictions.')

        z = X.dot(self._weights) + self._bias
        y_pred = self._sigmoid(z)
        return np.where(y_pred > 0.5, 1, 0)
