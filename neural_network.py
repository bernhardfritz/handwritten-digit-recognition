import random
import numpy as np
import json


class NeuralNetwork:
    def __init__(self, neurons_per_layer: list[int]):
        self.num_layers = len(neurons_per_layer)
        self.weights = [
            np.random.randn(n, n_1)
            for n_1, n in zip(neurons_per_layer[:-1], neurons_per_layer[1:])
        ]
        self.biases = [np.random.randn(n, 1) for n in neurons_per_layer[1:]]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def compute_gradient_descent_step(self, mini_batch, learning_rate):
        cost_weight = [np.zeros(w.shape) for w in self.weights]
        cost_bias = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_cost_weight, delta_cost_bias = self.backpropagation(x, y)
            cost_weight = [cw + dcw for cw, dcw in zip(cost_weight, delta_cost_weight)]
            cost_bias = [cb + dcb for cb, dcb in zip(cost_bias, delta_cost_bias)]
        learning_rate_per_batch = learning_rate / len(mini_batch)
        self.weights = [
            w - learning_rate_per_batch * cw for w, cw in zip(self.weights, cost_weight)
        ]
        self.biases = [
            b - learning_rate_per_batch * cb for b, cb in zip(self.biases, cost_bias)
        ]
        return

    def backpropagation(self, x, y):
        dc_dw = [np.zeros(w.shape) for w in self.weights]
        dc_db = [np.zeros(b.shape) for b in self.biases]
        # feedforward
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        # backward pass
        for i in range(1, self.num_layers):
            if i == 1:
                dc_da = self.cost_derivative(activations[-i], y)
            else:
                dz_da = self.weights[-i + 1].transpose()
                dc_da = np.dot(dz_da, dc_db[-i + 1])
            da_dz = sigmoid_prime(zs[-i])
            dz_db = 1
            dc_db[-i] = (dc_da * da_dz) * dz_db
            dz_dw = activations[-i - 1].transpose()
            dc_dw[-i] = np.dot(dc_db[-i], dz_dw)
        return dc_dw, dc_db

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, a, y):
        return 2 * (a - y)

    def train(
        self, training_data, epochs, mini_batch_length, learning_rate, test_data=None
    ):
        training_data_length = len(training_data)
        if test_data:
            test_data_length = len(test_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[i : i + mini_batch_length]
                for i in range(0, training_data_length, mini_batch_length)
            ]
            for mini_batch in mini_batches:
                self.compute_gradient_descent_step(mini_batch, learning_rate)
            if test_data:
                print(
                    "Epoch {}: {} / {}".format(
                        i, self.evaluate(test_data), test_data_length
                    )
                )
            else:
                print("Epoch {} complete".format(i))
        return

    def serialize(self, path):
        weights = [weight.tolist() for weight in self.weights]
        biases = [bias.tolist() for bias in self.biases]
        data = {"weights": weights, "biases": biases}
        with open(path, "w") as f:
            json.dump(data, f)

    def deserialize(path):
        with open(path, "r") as f:
            data = json.load(f)
        weights = data["weights"]
        biases = data["biases"]
        neurons_per_layer = [
            len(weights[0][0]),
            *[len(weight) for weight in weights],
        ]
        neural_network = NeuralNetwork(neurons_per_layer)
        neural_network.weights = [np.array(weight) for weight in weights]
        neural_network.biases = [np.array(bias) for bias in biases]
        return neural_network


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)
