import numpy as np

from NN_Visualizations import graph_smoothed_values

class NeuralNetworkLayer:

    def __init__(self, input_size, output_size, activation_function='sigmoid'):
        """

        :param input_size:
        :param output_size:
        :param activation_function: Name of the activation function, supported activation functions are:
        sigmoid
        relu
        """
        # Multiplying it by 2, then subtracting 1 lets the mean weight value be 0 instead of 0.5
        self.weights = 2 * np.random.random((input_size, output_size)) - 1
        self.bias = 2 * np.random.random(output_size) - 1

        self.input_size = input_size
        self.output_size = output_size

        activation_functions_dictionary = {'sigmoid': self.sigmoid,
                                           'relu': self.relu}

        self.activation_function = activation_functions_dictionary[activation_function]

    # Activation functions
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x)  # Note, the actual derivative is sig(x) * (1 - sig(x))
        return 1 / (1 + np.exp(x))

    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.greater(x, 0)
        return np.where(x > 0, x, 0)

    def feed_forward(self, input_values):
        output_values = self.weights.dot(self.weights)
        output_values += self.bias
        output_values = self.activation_function(output_values)

        return output_values

    def training_feed_forward(self, input_values):
        raw_output_values = input_values.dot(self.weights)
        raw_output_values += self.bias

        # Returns both activated and non-activated values.
        return self.activation_function(raw_output_values), raw_output_values


class NeuralNetwork:

    def __init__(self, layers: list[NeuralNetworkLayer], learning_rate=0.01):
        """

        :param layers: Iterable containing NeuralNetworkLayer objects. The objects are NOT copied.
        """
        # A list containing the layers, which contain the weights and biases of the network.
        self.layers = layers

        # Information about the network itself.
        self.input_size = self.layers[0].input_size
        self.output_size = self.layers[-1].output_size
        self.learning_rate = learning_rate

        # Diagnostic vars.

    def predict(self, input_values):

        out = input_values
        for layer in self.layers:
            out = layer.feed_forward(out)

        return out

    def epoch(self, input_values, desired_output_values, diagnostics=False):
        raw_outputs = []
        inputs = []
        # Feed forward part, to see how far off we are.
        out, raw_out = input_values, 0
        for layer in self.layers:
            inputs.append(out)
            out, raw_out = layer.training_feed_forward(out)
            raw_outputs.append(raw_out)

        error = 0.5 * (desired_output_values - out) ** 2  # MSE
        layer_error = out - desired_output_values
        # The backpropagation part
        for layer, raw_out, inp in zip(reversed(self.layers), reversed(raw_outputs), reversed(inputs)):
            delta = layer_error * layer.activation_function(raw_out, derivative=True)
            layer.bias += delta.sum(axis=0) * self.learning_rate
            layer.weights += inp.T.dot(delta) * self.learning_rate

        if diagnostics:
            return error


X = np.array([[0, 1],
              [1, 0],
              [0, 0],
              [1, 1]])

Y = np.array([[1],
              [1],
              [0],
              [0]])

if __name__ == "__main__":
    sample_network = NeuralNetwork([NeuralNetworkLayer(2, 3),
                                    NeuralNetworkLayer(3, 1)])

    errors = []
    for i in range(10000):
        if i % 1000 == 0:
            errors.append(sample_network.epoch(X, Y, True))
        sample_network.epoch(X, Y)

    graph_smoothed_values(errors)
