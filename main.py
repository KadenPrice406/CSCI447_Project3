import numpy as np
import pandas as pd
from copy import copy, deepcopy


class NeuralNetwork:

    def __init__(self, layer_sizes: list[int], use_softmax: bool = False, use_cel: bool = False):
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.hidden_sizes = layer_sizes[1:-1]
        rng = np.random.default_rng()
        self.weights = [rng.normal(loc=0, scale=0.1, size=(layer_sizes[i + 1], layer_sizes[i] + 1)) for i in
                        range(len(layer_sizes) - 1)]
        self.use_softmax = use_softmax
        self.use_cel = use_cel

    # Prediction Methods
    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x: np.ndarray[float]):
        total = np.sum(np.exp(x))
        z = np.array([np.exp(v) / total for i, v in enumerate(x)])
        return z

    @staticmethod
    def mse(p: np.ndarray[float], y: np.ndarray[float]):
        return np.linalg.norm(p - y, 2) ** 2 / p.shape[0]

    @staticmethod
    def cel(p: np.ndarray[float], y: np.ndarray[float]):
        return -np.sum(y * np.log(p))

    def predict(self, x: np.ndarray[float]):
        current_vector = deepcopy(x)
        pre_activations = []
        post_activations = []
        for i, w in enumerate(self.weights):
            current_vector = np.concatenate((current_vector, np.ones(shape=(1, 1))))
            post_activations.append(deepcopy(current_vector))
            current_vector = w @ current_vector
            pre_activations.append(deepcopy(current_vector))
            if i != len(self.weights) - 1:
                current_vector = self.logistic(current_vector)
            else:
                if self.use_softmax:
                    current_vector = self.softmax(current_vector)
        return current_vector, pre_activations, post_activations

    # Backprop Methods
    @staticmethod
    def D_mse(p: np.ndarray[float], y: np.ndarray[float]):
        return 2 / p.shape[0] * (p - y).T

    @staticmethod
    def D_cel(p: np.ndarray[float], y: np.ndarray[float]):
        return - (y / p).T

    @staticmethod
    def D_identity(a: np.ndarray[float]):
        return np.identity(a.shape[0])

    @staticmethod
    def D_softmax(a: np.ndarray[float]):
        D = np.ndarray(shape=(a.shape[0], a.shape[0]), dtype=float)
        sm_vals = NeuralNetwork.softmax(a)
        for i, v1 in enumerate(a):
            for j, v2 in enumerate(a):
                if i == j:
                    delta = 1
                else:
                    delta = 0
                D[i, j] = sm_vals[i, 0] * (delta - sm_vals[j, 0])
        return D

    @staticmethod
    def D_logistic(a: np.ndarray[float]):
        return np.diag((NeuralNetwork.logistic(a) * (1 - NeuralNetwork.logistic(a))).T[0])

    @staticmethod
    def D_prev_output(w: np.ndarray[float]):
        return w

    @staticmethod
    def D_weights(w: np.ndarray[float], o: np.ndarray[float]):
        return np.kron(o.T, np.identity(w.shape[0]))

    def backpropagation(self, x: np.ndarray[float], y: np.ndarray[float], eta: float):
        p, a, o = self.predict(x)
        if self.use_cel:
            D = self.D_cel(p, y)
        else:
            D = self.D_mse(p, y)
        for i in range(len(self.weights) - 1, -1, -1):
            # For current Layer
            if i == len(self.weights) - 1:
                if self.use_softmax:
                    update = self.D_softmax(a[i])
                else:
                    update = self.D_identity(a[i])
            else:
                update = self.D_logistic(a[i])
            update = update @ self.D_weights(self.weights[i], o[i])
            self.weights[i] -= eta * np.reshape(D @ update, self.weights[i].shape)

            # For Backpropagation
            if i == len(self.weights) - 1:
                if self.use_softmax:
                    D = D @ self.D_softmax(a[i])
                else:
                    D = D @ self.D_identity(a[i])
            else:
                D = D @ self.D_logistic(a[i])
            D = (D @ self.D_prev_output(self.weights[i]))[:, :-1]
        if self.use_cel:
            return self.cel(self.predict(x)[0], y)
        else:
            return self.mse(self.predict(x)[0], y)


if __name__ == '__main__':
    AE = NeuralNetwork([2, 3, 3], use_softmax=False, use_cel=False)
    predict = AE.predict(np.array([[1],
                                   [0]]))
    mse = AE.mse(predict[0], np.array([[0],
                                       [1],
                                       [0]]))
    print(mse)
    min_mse = np.inf
    for _ in range(10000):
        test = AE.backpropagation(np.array([[1],
                                            [0]]),
                                  np.array([[0],
                                            [1],
                                            [0]]),
                                  0.01)
        if test < min_mse:
            min_mse = test
    print(min_mse)

