from layer import Layer
import numpy as np
from weights import (
    he_normal,
    he_uniform,
    xavier_normal,
    xavier_uniform,
    normal,
    uniform,
)


# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, w_init=xavier_uniform):
        self.input = None
        # self.weights = np.random.rand(input_size, output_size) - 0.5
        self.weights = w_init(input_size, output_size)

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        return input_error
