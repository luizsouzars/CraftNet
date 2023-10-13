import numpy as np


class Network:
    """
    A class representing a simple neural network.

    Attributes:
        layers (list): List of layers in the network.
        loss: Loss function used to calculate the error.
        loss_prime: Derivative of the loss function used for training.
    """

    def __init__(self):
        """
        Initializes a new instance of the Network class.
        """
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        """
        Adds a layer to the network.

        Parameters:
            layer: An instance of a neural layer.
        """
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        """
        Sets the loss function and its derivative for the network.

        Parameters:
            loss: Loss function to be used for calculating the error.
            loss_prime: Derivative of the loss function used for training.
        """
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data: np.ndarray):
        """
        Performs output prediction for provided input data.

        Parameters:
            input_data: Input data for which predictions will be made.

        Returns:
            A list containing the predicted outputs for each input.
        """
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer,
    ):
        """
        Trains the neural network using the provided training dataset.

        Parameters:
            x_train: Training input data.
            y_train: Labels corresponding to the training input data.
            epochs: Number of training epochs.
            batch_size: Size of mini-batches used during training.
            optimizer: An instance of an optimizer.
        """

        # sample dimension first
        samples = len(x_train)
        iterations = samples // batch_size

        # training loop
        for i in range(epochs):
            err_epoch = 0

            # mini-batches
            rng = np.random.default_rng()
            mini_batches = []

            for _ in range(iterations):
                mini_batches.append(rng.choice(samples, batch_size, replace=False))

            if samples % batch_size != 0:
                remainder = samples - (iterations * batch_size)
                mini_batches.append(rng.choice(samples, remainder, replace=False))

            for iterarion, mini_batch in enumerate(mini_batches):
                err_iteration = 0
                for j in mini_batch:
                    # forward propagation
                    output = x_train[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)

                    # compute loss (for display purpose only)
                    err_iteration += self.loss(y_train[j], output)
                    err_epoch += err_iteration

                    # backward propagation
                    error = self.loss_prime(y_train[j], output)
                    for layer in reversed(self.layers):
                        error = layer.backward_propagation(
                            error, optimizer.learning_rate
                        )
                print(
                    f"epoch:{i+1}    iteration:{iterarion+1}/{iterations}   error={err_iteration}"
                )

            # calculate average error on all samples
            err_epoch /= samples
            print("epoch %d/%d   error=%f" % (i + 1, epochs, err_epoch))
