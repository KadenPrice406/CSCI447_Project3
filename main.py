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


class Autoencoder:

    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.05):
        if hidden_dim >= input_dim:
            raise ValueError("hidden_dim must be strictly smaller than input_dim")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.model = NeuralNetwork([input_dim, hidden_dim, input_dim], use_softmax=False, use_cel=False)

    def encode(self, x: np.ndarray[float]) -> np.ndarray[float]:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        h_in = np.concatenate((x, np.ones((1, 1))), axis=0)
        z = self.model.weights[0] @ h_in
        return self.model.logistic(z)

    def reconstruct(self, x: np.ndarray[float]) -> np.ndarray[float]:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        h = self.encode(x)
        h_with_bias = np.concatenate((h, np.ones((1, 1))), axis=0)
        return self.model.weights[1] @ h_with_bias

    def train(self, X: np.ndarray[float], epochs: int = 1000):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        for _ in range(epochs):
            for row in X:
                x = row.reshape(-1, 1)
                self.model.backpropagation(x, x, self.learning_rate)


def train_autoencoder(X: np.ndarray[float], hidden_dim: int, epochs: int = 1000, learning_rate: float = 0.05):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    input_dim = X.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, learning_rate=learning_rate)
    autoencoder.train(X, epochs=epochs)
    return autoencoder


if __name__ == '__main__':
    X_train = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ])

    AE = Autoencoder(input_dim=2, hidden_dim=1, learning_rate=0.001)
    initial_reconstructions = np.array([AE.reconstruct(x).ravel() for x in X_train])
    initial_mse = np.mean((initial_reconstructions - X_train) ** 2)
    print("Initial reconstruction MSE:", initial_mse)

    AE.train(X_train, epochs=1000)

    reconstructions = np.array([AE.reconstruct(x).ravel() for x in X_train])
    final_mse = np.mean((reconstructions - X_train) ** 2)
    print("Final reconstruction MSE:", final_mse)
    print("Encoded first sample:")
    print(AE.encode(X_train[0]))
    print("Reconstruction of first sample:")
    print(AE.reconstruct(X_train[0]))
