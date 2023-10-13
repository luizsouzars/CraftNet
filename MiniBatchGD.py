import numpy as np


def mini_batch(x: np.ndarray, y: np.ndarray, batch_size: int):
    print("x shape", x.shape)
    print("y shape", y.shape)
    batches = []
    data = np.hstack((x.reshape((x.shape[0], x.shape[1] * x.shape[2])), y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size

    for i in range(n_minibatches + 1):
        mb = data[i * batch_size : (i + 1) * batch_size, :]
        X_mini = mb[:, :-1].reshape((x.shape[0], x.shape[1], x.shape[2]))
        Y_mini = mb[:, -1].reshape((-1, 1))
        batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mb = data[i * batch_size : data.shape[0]]
        X_mini = mb[:, :-1].reshape((x.shape[0], x.shape[1], x.shape[2]))
        Y_mini = mb[:, -1].reshape((-1, 1))
        batches.append((X_mini, Y_mini))
    return batches
