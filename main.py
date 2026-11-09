import numpy as np
import pandas as pd
from copy import copy, deepcopy


class NeuralNetwork:

    def __init__(self, layer_sizes: list[int], use_softmax: bool = False, use_cel: bool = False):
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        self.hidden_sizes = layer_sizes[1:-1]
        rng = np.random.default_rng()
        self.weights = [rng.normal(loc=0, scale=0.1, size=(layer_sizes[i+1], layer_sizes[i]+1)) for i in range(len(layer_sizes) - 1)]
        self.use_softmax = use_softmax
        self.use_cel = use_cel

    @staticmethod
    def sigmoid(x):
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
        for i, w in enumerate(self.weights):
            current_vector = np.concatenate((current_vector, np.ones(shape=(1, 1))))
            current_vector = w @ current_vector
            if i != len(self.weights) - 1:
                current_vector = self.sigmoid(current_vector)
            else:
                if self.use_softmax:
                    current_vector = self.softmax(current_vector)
        return current_vector


if __name__ == '__main__':
    AE = NeuralNetwork([2, 1, 2], use_softmax=False, use_cel=False)
    predict = AE.predict(np.zeros(shape=(2, 1)))
    mse = AE.mse(predict, np.array([[1],
                                    [0]]))
    print(mse)

