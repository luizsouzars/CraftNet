import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(yhat: np.ndarray, y: np.ndarray) -> float:
    """Compute binary cross-entropy loss for a vector of predictions

    Parameters
    ----------
    yhat
        An array with len(yhat) predictions between [0, 1]
    y
        An array with len(y) labels where each is one of {0, 1}
    """
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()
