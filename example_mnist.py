import numpy as np
import idx2numpy
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from MiniBatchGD import mini_batch
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
    softmax,
    softmax_prime,
)
from losses import mse, mse_prime
from optimizers import adam, sgd
from weights import (
    he_normal,
    he_uniform,
    normal,
    uniform,
    xavier_normal,
    xavier_uniform,
)

train_img = idx2numpy.convert_from_file(r"./mnist/train-images.idx3-ubyte").reshape(
    60000, 1, 28 * 28
)
train_lbs = idx2numpy.convert_from_file(r"./mnist/train-labels.idx1-ubyte").reshape(
    -1, 1
)

test_img = idx2numpy.convert_from_file(r"./mnist/t10k-images.idx3-ubyte").reshape(
    10000, 1, 28 * 28
)
test_lbs = idx2numpy.convert_from_file(r"./mnist/t10k-labels.idx1-ubyte").reshape(-1, 1)

print(train_img.shape)
print(train_lbs.shape)
print(test_img.shape)
print(test_lbs.shape)

# Normalização dos valores entre [0,1]
train_img = train_img.astype("float32") / 255
test_img = test_img.astype("float32") / 255


# One Hot Encoding Labels
def OHE(y_lbs: np.ndarray, num_classes: int):
    ohs = np.zeros((y_lbs.shape[0], num_classes))
    for i, j in enumerate(y_lbs):
        ohs[i][j] = 1
    return ohs


train_lbs = OHE(train_lbs, 10)
test_lbs = OHE(test_lbs, 10)

# # Implementação da rede
net = Network()
net.add(FCLayer(28 * 28, 100))
net.add(ActivationLayer(sigmoid, sigmoid_prime))
net.add(FCLayer(100, 50, he_normal))
net.add(ActivationLayer(leakyReLU, leakyReLU_prime))
net.add(FCLayer(50, 10, uniform))
net.add(ActivationLayer(ReLU, ReLU_prime))

# train
net.use(mse, mse_prime)
net.fit(train_img, train_lbs, epochs=5, batch_size=64, optimizer=adam())

# test
out = net.predict(test_img)
# print(out)


for i in range(len(test_lbs[:20])):
    print("y    :", np.argmax(test_lbs[i]))
    print("y_hat:", np.argmax(out[i]))
    print()
