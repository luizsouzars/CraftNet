import numpy as np

"""activation function and its derivative"""


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    res = 1 / (1 + np.exp(-x))
    return res


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    res = sigmoid(x) * (1 - sigmoid(x))
    return res


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def ReLU_prime(x: np.ndarray) -> np.ndarray:
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def leakyReLU(x: np.ndarray, alpha=0.01) -> np.ndarray:
    return np.maximum(alpha * x, x)


def leakyReLU_prime(x, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, 1, alpha)


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    return 1

def softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def softmax_prime(x: np.ndarray, y_oh: np.ndarray) -> np.ndarray:
    y_pred = softmax(x)
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    y_pred[k] = pk * (1.0 - pk)
    return y_pred
