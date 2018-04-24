class MFHyperparameters:
    def __init__(self, d, gamma_array):
        """
        Bn Bias
        U users matrix
        V items matrix
        m users index
        n items index
        """
        self.gamma_array = gamma_array
        self.d = d


class MFALSHyperparameters(MFHyperparameters):
    def __init__(self, d, gamma_array, epsillon):
        super(MFALSHyperparameters, self).__init__(d, gamma_array)
        self.epsillon = epsillon

    def __str__(self):
        return "Hyperparameters:\n" \
               "----------------\n" \
               " - D        = {}\n" \
               " - Lamda    = {}\n" \
               " - Epsillon = {}\n\n" \
               "".format(self.d, self.gamma_array, self.epsillon)

class MFSGDHyperparameters(MFHyperparameters):
    def __init__(self, d, gamma_array, alpha, epochs):
        super(MFSGDHyperparameters, self).__init__(d, gamma_array)
        self.epochs = epochs
        self.alpha = alpha

    def __str__(self):
        return "Hyperparameters:\n" \
               "----------------\n" \
               " - D        = {}\n" \
               " - Lamda    = {}\n" \
               " - alpha    = {}\n" \
               " - epochs   = {}\n\n" \
               "".format(self.d, self.gamma_array, self.alpha, self.epochs)