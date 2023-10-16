import numpy as np


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculate the accuracy of predicted values.

    Parameters:
    - y (numpy.ndarray): True labels.
    - y_hat (numpy.ndarray): Predicted labels.

    Returns:
    - float: Accuracy of the predictions.
    """

    # Ensure the input arrays are numpy arrays
    y = np.asarray([np.argmax(num) for num in y])
    y_hat = np.asarray([np.argmax(num) for num in y_hat])

    # Get the number of samples
    N = y.shape[0]

    # Calculate accuracy
    acc = (y == y_hat).sum() / N

    return acc


# Example Usage:
# true_labels = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
# predicted_labels = np.array([[0.1, 0.2, 0.7], [0.3, 0.6, 0.1], [0.8, 0.1, 0.1]])
# acc = accuracy(true_labels, predicted_labels)
# print(f"Accuracy: {acc}")
