import numpy as np


class adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Inicializa o otimizador Adam com os parâmetros específicos.

        Parâmetros:
        - learning_rate (float): Taxa de aprendizado para controlar o tamanho dos passos de atualização.
        - beta1 (float): Coeficiente de decaimento para o momento de primeira ordem.
        - beta2 (float): Coeficiente de decaimento para o momento de segunda ordem.
        - epsilon (float): Valor pequeno adicionado para evitar a divisão por zero.

        Atributos:
        - learning_rate (float): Taxa de aprendizado.
        - beta1 (float): Coeficiente de decaimento para o momento de primeira ordem.
        - beta2 (float): Coeficiente de decaimento para o momento de segunda ordem.
        - epsilon (float): Valor pequeno adicionado para evitar a divisão por zero.
        - moment1 (array): Momento de primeira ordem (média móvel) dos gradientes.
        - moment2 (array): Momento de segunda ordem (média móvel ponderada dos quadrados) dos gradientes.
        - timestep (int): Número de iterações até o momento.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = None
        self.moment2 = None
        self.timestep = 0

    def update(self, gradients):
        """
        Realiza uma iteração de atualização dos pesos usando o otimizador Adam.

        Parâmetros:
        - gradients (array): Gradientes da função de custo em relação aos pesos.

        Retorna:
        - update (array): Atualização dos pesos calculada pelo otimizador.
        """

        self.timestep += 1

        # Inicialização dos momentos
        if (self.moment1 is None) or (self.moment1.shape != gradients.shape):
            self.moment1 = np.zeros_like(gradients)
            self.moment2 = np.zeros_like(gradients)

        # Atualização dos momentos
        self.moment1 = self.beta1 * self.moment1 + (1 - self.beta1) * gradients
        self.moment2 = self.beta2 * self.moment2 + (1 - self.beta2) * (gradients**2)

        # Correção de viés dos momentos
        moment1_hat = self.moment1 / (1 - self.beta1**self.timestep)
        moment2_hat = self.moment2 / (1 - self.beta2**self.timestep)

        # Atualização dos pesos usando os momentos corrigidos
        update = (
            -self.learning_rate * moment1_hat / (np.sqrt(moment2_hat) + self.epsilon)
        )
        return update


class sgd:
    def __init__(self, learning_rate, momentum, nesterov_momentum):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov_momentum = nesterov_momentum

    def update(self):
        None
