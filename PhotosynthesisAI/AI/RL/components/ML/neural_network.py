from abc import ABC
from typing import Tuple, Dict, Any

import numpy as np


class ActFns:
    @staticmethod
    def sigmoid(x, differential=False):
        if differential:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x, differential=False):
        if differential:
            return 1 - np.power(x, 2)
        else:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def relu(x, differential=False):
        if differential:
            return np.where(x > 0, 1, 0)
        else:
            return np.where(x >= 0, x, 0)

    @staticmethod
    def identity(x, differential=False):
        if differential:
            return 1
        else:
            return x


class NeuralNetwork(ABC):
    def __init__(
        self,
        layer_sizes: Tuple,
        learning_rate: float = 0.05,
        hidden_activation_function: Any = ActFns.relu,
        output_activation_function: Any = ActFns.sigmoid,
        weights: Dict[int, np.array] = None,
        bias: Dict[int, np.array] = None,
        seed: int = None,
    ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function
        self.weights = weights
        self.bias = bias
        self.seed = seed

    def _initialize_parameters(self, number_of_input_features: int):
        if self.seed:
            np.random.seed(self.seed)
        layer_sizes = [number_of_input_features] + list(self.layer_sizes)
        self.weights = {
            i + 1: np.random.randn(layer_size, layer_sizes[i]) * 0.01 for i, layer_size in enumerate(layer_sizes[1:])
        }
        self.bias = {i + 1: np.zeros((layer_size, 1)) * 0.01 for i, layer_size in enumerate(self.layer_sizes)}

    def _calculate_layer_output(self, W, X, b):
        return np.dot(W, X) + b

    def _calculate_hidden_layer(self, W, X, b):
        Z = self._calculate_layer_output(W, X, b)
        return self.hidden_activation_function(Z)

    def _calculate_output_layer(self, W, X, b):
        Z = self._calculate_layer_output(W, X, b)
        return self.output_activation_function(Z)

    def _forward_propagation(self, X):
        A = {0: X}
        num_layers = len(self.layer_sizes)

        # calculate hidden layers
        for layer_number in range(1, num_layers):
            A[layer_number] = self._calculate_hidden_layer(
                W=self.weights[layer_number], X=A[layer_number - 1], b=self.bias[layer_number]
            )

        # calculate output layer
        A[num_layers] = self._calculate_output_layer(W=self.weights[num_layers], X=A[num_layers - 1], b=self.bias[num_layers])
        return A

    def _compute_cost(self, A, Y):
        pass

    def _get_differential_cost_function(self, A, Y):
        pass

    def _backward_propagation(self, A, Y):
        num_layers = len(self.layer_sizes)
        m = Y.shape[0]
        dZ = {}
        dW = {}
        db = {}
        for layer_number in range(num_layers, 0, -1):
            if layer_number == num_layers:
                # calculate final layer first - this relies on cost function, not on layer after
                # dl/dz = dl/da * da/dz = differential of cost function * differential of activation function
                dZ[layer_number] = self._get_differential_cost_function(A, Y) * self.output_activation_function(
                    A[layer_number], differential=True
                )
            else:

                dZ[layer_number] = np.dot(
                    self.weights[layer_number + 1].T, dZ[layer_number + 1]
                ) * self.hidden_activation_function(A[layer_number], differential=True)
            dW[layer_number] = (1 / m) * np.dot(dZ[layer_number], A[layer_number - 1].T)
            db[layer_number] = (1 / m) * np.sum(dZ[layer_number], axis=1, keepdims=True)
        return dW, db

    def _update_parameters(self, weight_updates, bias_updates):
        for i, update in weight_updates.items():
            self.weights[i] -= update * self.learning_rate
        for i, update in bias_updates.items():
            self.bias[i] -= update * self.learning_rate

    def fit(self, X, Y, num_iterations=1, verbose=False):
        X = np.array(X)
        Y = np.array(Y).reshape(1, X.shape[1])
        if (not self.weights) or (not self.bias):
            self._initialize_parameters(number_of_input_features=X.shape[0])
        for i in range(num_iterations):
            A = self._forward_propagation(X)
            cost = self._compute_cost(A, Y)
            if verbose and (i % 1000 == 0):
                print(f"Cost after {i} iterations: {cost}")
            weight_updates, bias_updates = self._backward_propagation(A, Y)
            self._update_parameters(weight_updates, bias_updates)

    def predict(self):
        pass

    def reset(self):
        self.weights = self.bias = None


if __name__ == "__main__":
    X = np.array([[1.62434536, -0.61175641, -0.52817175], [-1.07296862, 0.86540763, -2.3015387]])
    Y = np.array([[0, 1, 0]])
    nn = NeuralNetwork(layer_sizes=(4, 1), seed=3)
    nn.fit(X, Y, verbose=True)
    predictions = nn.predict(X)


class NNClassifier(NeuralNetwork):

    def _compute_cost(self, A, Y):
        final_layer = len(self.layer_sizes)
        m = Y.shape[1]
        logprobs = np.multiply(Y, np.log(A[final_layer])) + np.multiply(1 - Y, np.log(1 - A[final_layer]))
        cost = -(1 / m) * np.sum(logprobs)
        if not cost > 0:
            raise ValueError(f'Cost is not positive. Final output is {A[final_layer]}')
        return cost

    def _get_differential_cost_function(self, A, Y):
        final_layer = len(self.layer_sizes)
        return (A[final_layer] - Y) / (A[final_layer] * (1 - A[final_layer]))

    def predict(self, X):
        X = np.array(X)
        final_layer = len(self.layer_sizes)
        A = self._forward_propagation(X)
        predictions = np.where(A[final_layer] > 0.5, 1, 0)
        return np.squeeze(predictions)

    def predict_proba(self, X):
        X = np.array(X)
        final_layer = len(self.layer_sizes)
        A = self._forward_propagation(X)
        predictions = A[final_layer]
        return np.squeeze(predictions).reshape(1, X.shape[1])


class NNRegressor(NeuralNetwork):

    def _compute_cost(self, A, Y):
        final_layer = len(self.layer_sizes)
        m = Y.shape[1]
        cost = 0.5 * np.sum(np.power(Y - A[final_layer], 2), axis=1, keepdims=True)
        if not cost >= 0:
            raise ValueError(f'Cost is not positive. Final output is {A[final_layer]}')
        return cost

    def _get_differential_cost_function(self, A, Y):
        final_layer = len(self.layer_sizes)
        return -1 * 0.5 * np.sum(Y - A[final_layer], axis=1, keepdims=True)

    def predict(self, X):
        X = np.array(X)
        final_layer = len(self.layer_sizes)
        A = self._forward_propagation(X)
        predictions = A[final_layer]
        return np.squeeze(predictions).reshape(1, X.shape[1])


