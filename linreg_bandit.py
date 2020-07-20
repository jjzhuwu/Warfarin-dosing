import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from bandit import Bandit

class LinReg_Bandit(Bandit):

    def __init__(self, threshold, reward, alpha=1.0):

        """
        Requires: reward has shape (len(threshold)+1, len(threshold)+1)
        """

        self.alpha = alpha
        self.r = reward
        self.threshold = threshold
        self.model_name = "LinReg"

    def data_load(self, X, y, shuffle=True):

        self.X = X.values
        self.y = y.values

        self.T = X.shape[0]
        self.d = X.shape[1]

        if shuffle:
            indexes = np.random.permutation(self.T)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes]

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def run(self, initial_guess=0):

        self.accurate_so_far = np.zeros(self.T)
        self.regret_history = np.zeros(self.T)

        correct_labels = 0
        regret = 0

        for i in range(self.T):
            if i < self.d:
                y_pred = initial_guess
            else:
                model = Ridge(alpha=self.alpha)
                model.fit(self.X[0:i, :], self.y[0:i])
                y_pred = model.predict(self.X[i, :].reshape(1, -1))[0]
                y_pred_choice = np.sum(y_pred >= self.threshold)
                y_correct_choice = np.sum(self.y[i] >= self.threshold)

                correct_labels += y_pred_choice == y_correct_choice
                regret += np.max(self.r[y_correct_choice, :]) - self.r[y_correct_choice, y_pred_choice]

                self.accurate_so_far[i] = correct_labels
                self.regret_history[i] = regret
