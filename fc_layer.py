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
        self.weights = w_init(input_size, output_size)

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights)
        return self.output

    # computes dE/dW for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, optimizer):
        # derivada parcial da função de erro em relação à camada de entrada.
        input_error = np.dot(output_error, self.weights.T)
        # Gradientes
        gradients = np.dot(self.input.T, output_error)
        # update = optimizer.update(gradients)

        # update parameters
        self.weights -= optimizer.learning_rate * gradients
        # self.weights += update
        return input_error
