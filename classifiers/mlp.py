# third-party imports
import numpy as np


class CustomMLPClassifier:
    def __init__(self, num_hidden_layers=5):
        self._fitted = False
        self._errors = list()
        self._num_hidden_layers = num_hidden_layers

    def __init_params(self, X, y):
        self._w1 = 2 * np.random.random((len(X[0]), self._num_hidden_layers)) - 1
        self._w2 = 2 * np.random.random((self._num_hidden_layers, len(y[0]))) - 1

    def _forward_propagation(self, X):
        # activate the first layer using the input
        l1 = 1 / (1 + np.exp(-(np.dot(X, self._w1))))
        # activate the second layer using first layer as input
        l2 = 1 / (1 + np.exp(-(np.dot(l1, self._w2))))

        return l1, l2

    def _backward_propagation(self, X, y, l1, l2, lr):
        # find contribution of error on each weight on the second layer
        l2_delta = (y - l2) * (l2 * (1 - l2))
        # update each weight in the second layer slowly
        self._w2 += l1.T.dot(l2_delta) * lr

        # find contribution of error on each weight on the second layer w.r.t the first layer
        l1_delta = l2_delta.dot(self._w2.T) * (l1 * (1-l1))
        # udpate weights in the first layer
        self._w1 += X.T.dot(l1_delta) * lr

    def fit(self, X, y, epochs=1000, lr=0.05, verbose=False):

        self.__init_params(X, y)

        for _ in range(epochs):
            l1, l2 = self._forward_propagation(X)

            # find the average error of this batch
            er = (abs(y - l2)).mean()
            self._errors.append(er)

            self._backward_propagation(X, y, l1, l2, lr)

            if verbose:
                print('Error:', er)

        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            raise Exception('Please call `SimpleNeuralNetwork.fit(X, y)` before making predictions.')

        l1 = 1 / (1 + np.exp(-(np.dot(X, self._w1))))
        l2 = 1 / (1 + np.exp(-(np.dot(l1, self._w2))))

        return np.argmax(l2, axis=1)
