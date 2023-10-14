#from layer import Layer
import activations
from weights import Xavier, He
from optimizers import adam, sgd
import numpy as np

# inherit from base class Layer
class FCLayer():
    # input_size = number of input neurons
    # output_size = number of output neurons
    # activation = activation function
    def __init__(self, 
                 input_size, 
                 output_size, 
                 activation, 
                 activation_prime, 
                 initilization=Xavier,
                 optimizer=sgd):
        
        self.weights = np.random.rand(input_size, output_size) - 0.5
        # self.bias = np.random.rand(1, output_size).reshape((-1, output_size)) #Implementação original
        self.activation = activation
        self.activation_prime = activation_prime
        self.optimizer = optimizer
        self.bias = 1  # Inicialização fixa em 1

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        function_act_value = activations.self.activation(self.output)
        return function_act_value

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
