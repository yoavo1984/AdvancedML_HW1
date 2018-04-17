class MFHyperparameters:
    def __init__(self, k, alpha, gamma_array):
        """
        Bn Bias
        U users matrix
        V items matrix
        m users index
        n items index
        """
        self.gamma_array = gamma_array
        self.k = k
        self.alpha = alpha


class MFALSHyperparameters(MFHyperparameters):
    def __init__(self, k, alpha, gamma_array, epsilon):
        super(MFALSHyperparameters, self).__init__(k, alpha, gamma_array)
        self.epsilon = epsilon


class MFSGDHyperparameters(MFHyperparameters):
    def __init__(self, k, alpha, gamma_array, epochs):
        super(MFSGDHyperparameters, self).__init__(k, alpha, gamma_array)
        self.epochs = epochs
