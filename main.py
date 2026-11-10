import numpy as np
import pandas as pd
from copy import copy, deepcopy
from pathlib import Path


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


class DatasetLoaderPreprocessor:

    def __init__(self, data_dir: str = "."):
        script_dir = Path(__file__).resolve().parent
        requested_dir = Path(data_dir)
        self.data_dir = requested_dir if requested_dir.is_absolute() else script_dir / requested_dir
        self.class_names: dict[str, list[str]] = {}
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None

    def _read_data(self, filename: str, header: int | None = None) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return pd.read_csv(path, header=header, dtype=str)

    @staticmethod
    def _numeric_features(frame: pd.DataFrame) -> np.ndarray:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        medians = numeric.median()
        numeric = numeric.fillna(medians).fillna(0.0)
        return numeric.to_numpy(dtype=float)

    @staticmethod
    def _categorical_features(
        frame: pd.DataFrame,
        categories: dict[str, list[str]],
    ) -> np.ndarray:
        encoded_columns = []
        for column, values in categories.items():
            invalid_values = set(frame[column].dropna()) - set(values)
            if invalid_values:
                raise ValueError(f"Unexpected values in {column}: {sorted(invalid_values)}")
            categorical = pd.Series(pd.Categorical(frame[column], categories=values), name=column)
            encoded_columns.append(pd.get_dummies(categorical, dtype=float))
        return pd.concat(encoded_columns, axis=1).to_numpy(dtype=float)

    def _classification_target(self, labels: pd.Series, dataset_name: str) -> np.ndarray:
        labels = labels.astype(str)
        classes = sorted(labels.unique())
        self.class_names[dataset_name] = classes
        class_indices = {label: index for index, label in enumerate(classes)}
        indices = labels.map(class_indices).to_numpy(dtype=int)
        return np.eye(len(classes), dtype=float)[indices]

    @staticmethod
    def _regression_target(labels: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(labels, errors="coerce")
        if numeric.isna().any():
            raise ValueError("Regression target contains missing or non-numeric values")
        return numeric.to_numpy(dtype=float).reshape(-1, 1)

    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Fit standardization on training features and return their scaled values."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D feature array")
        self.feature_mean = X.mean(axis=0)
        self.feature_scale = X.std(axis=0)
        self.feature_scale[self.feature_scale == 0.0] = 1.0
        return self.transform_features(X)

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize features using statistics fitted on training data."""
        if self.feature_mean is None or self.feature_scale is None:
            raise ValueError("Call fit_transform_features on training data first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("X must be 2D and have the same feature count used during fitting")
        return (X - self.feature_mean) / self.feature_scale

    def load_breast_cancer(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("breast-cancer-wisconsin.data")
        if data.shape[1] != 11:
            raise ValueError("Breast cancer data should have 10 features and a class column")
        X = self._numeric_features(data.iloc[:, 1:10])
        y = self._classification_target(data.iloc[:, 10], "breast_cancer")
        return X, y

    def load_car_evaluation(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("car.data")
        if data.shape[1] != 7:
            raise ValueError("Car evaluation data should have 6 features and a class column")
        features = data.iloc[:, :6].copy()
        features.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
        X = self._categorical_features(features, {
            "buying": ["vhigh", "high", "med", "low"],
            "maint": ["vhigh", "high", "med", "low"],
            "doors": ["2", "3", "4", "5more"],
            "persons": ["2", "4", "more"],
            "lug_boot": ["small", "med", "big"],
            "safety": ["low", "med", "high"],
        })
        y = self._classification_target(data.iloc[:, 6], "car_evaluation")
        return X, y

    def load_congressional_votes(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("house-votes-84.data")
        if data.shape[1] != 17:
            raise ValueError("Congressional votes data should have a class and 16 vote columns")
        labels = data.iloc[:, 0]
        votes = data.iloc[:, 1:17].replace("?", "unknown")
        votes.columns = [f"vote_{i}" for i in range(votes.shape[1])]
        X = self._categorical_features(votes, {
            column: ["y", "n", "unknown"] for column in votes.columns
        })
        y = self._classification_target(labels, "congressional_votes")
        return X, y

    def load_abalone(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("abalone.data")
        if data.shape[1] != 9:
            raise ValueError("Abalone data should have 8 features and a rings target")
        sex = self._categorical_features(
            data.iloc[:, :1].set_axis(["sex"], axis=1),
            {"sex": ["M", "F", "I"]},
        )
        measurements = self._numeric_features(data.iloc[:, 1:8])
        X = np.concatenate((sex, measurements), axis=1)
        y = self._regression_target(data.iloc[:, 8])
        return X, y

    def load_computer_hardware(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("machine.data")
        if data.shape[1] != 10:
            raise ValueError("Computer hardware data should have 10 columns")
        # Vendor, model name, and ERP are excluded per machine.names; PRP is the target.
        X = self._numeric_features(data.iloc[:, 2:8])
        y = self._regression_target(data.iloc[:, 8])
        return X, y

    def load_forest_fires(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._read_data("forestfires.data", header=0)
        expected_columns = {
            "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI",
            "temp", "RH", "wind", "rain", "area",
        }
        if set(data.columns) != expected_columns:
            raise ValueError("Forest fires data does not have the expected columns")
        categorical = self._categorical_features(data[["month", "day"]], {
            "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            "day": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
        })
        numeric = self._numeric_features(data[["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]])
        X = np.concatenate((numeric, categorical), axis=1)
        y = self._regression_target(data["area"])
        return X, y


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
