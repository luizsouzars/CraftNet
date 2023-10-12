
class adam():
    
    def __init__(self, learning_rating, n_steps, alpha, beta, gamma, epsilon):
        self.learning_rating = learning_rating
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self):
        None


class sgd():
    
    def __init__(self, learning_rating, momentum, nesterov_momentum):
        self.learning_rating = learning_rating
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum
        
    def update(self):
        None