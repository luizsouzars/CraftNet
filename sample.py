import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import (
    tanh,
    tanh_prime,
    sigmoid,
    sigmoid_prime,
    ReLU,
    ReLU_prime,
    leakyReLU,
    leakyReLU_prime,
    linear,
    linear_prime,
)
from losses import mse, mse_prime
from optimizers import adam, sgd

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 64))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(64, 32))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(32, 32))
net.add(ActivationLayer(ReLU, ReLU_prime))
net.add(FCLayer(32, 16))
net.add(ActivationLayer(leakyReLU, leakyReLU_prime))
net.add(FCLayer(16, 1))
net.add(ActivationLayer(linear, linear_prime))

# train
net.use(mse, mse_prime)
net.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=len(x_train),
    optimizer=adam(learning_rate=0.01, alpha=None, beta=0.9, gamma=0.99, epsilon=1e-20),
)

# test
out = net.predict(x_train)
print(out)
