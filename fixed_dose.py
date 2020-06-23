import numpy as np

class Fixed_Dose:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.array(["Medium" for i in range(X.shape[0])])
