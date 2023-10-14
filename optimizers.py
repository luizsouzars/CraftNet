
class adam:
    def __init__(
        self,
        learning_rate=0.01,
        n_steps=10,
        alpha=None,
        beta=0.9,
        gamma=0.99,
        epsilon=1e-20,
    ):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self):
        None

class sgd:
    def __init__(self, learning_rate, momentum, nesterov_momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum

    def update(self):
        None
