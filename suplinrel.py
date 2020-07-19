import numpy as np
from linrel import LinRel

class SupLinRel(LinRel):

    def __init__(self, K, reward, delta=0.1):

        """
        Parameters:
            K: number of actions
            reward: matrix of size K*K, reward[i][j] gives the deterministic
                reward when the true label is i and the prediction is j
        """
        self.K = K
        self.A_s = np.array(range(K))

        self.orig_reward = reward
        self.r = self.normalize_reward(reward)

        self.delta = delta

        self.model_name = "SupLinRel"

    def data_load(self, X, y, shuffle=True):

        self.X = X.values
        self.y = y

        X_max = np.max(X, axis=0)
        self.Z_max = np.append(X_max, np.ones(self.K-1))

        self.T = X.shape[0]
        self.d = X.shape[1]
        self.Z = np.zeros((self.d+self.K-1, 0))
        self.past_reward = np.zeros(0)

        if shuffle:
            indexes = np.random.permutation(self.T)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes]

    def run(self):

        regret = 0
        correct_labels = 0

        self.regret_history = np.zeros(self.T)
        self.accurate_so_far = np.zeros(self.T)

        s=1
        for j in range(self.T):

            ucb, width = self.compute_ucb(j)

            while np.sum(width > np.power(2.0, -s)) == 0 and np.sum(width > 1/np.sqrt(self.T)) > 0:
                self.Z = np.zeros((self.d+self.K-1, 0))
                self.past_reward = np.zeros(0)
                s += 1
                self.A_s = self.A_s[ucb >= np.max(ucb)-2*np.power(2.0, -s)]
                ucb, width = self.compute_ucb(j)

            if self.Z.shape[1] < self.d:
                choice = self.A_s[np.random.randint(len(self.A_s))]

                regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = regret
                self.accurate_so_far[j] = correct_labels

                self.Z = np.append(self.Z, self.create_zi(self.X[j, :], choice).reshape(-1, 1), axis=1)
                self.past_reward = np.append(self.past_reward, self.r[self.y[j], choice])

            elif np.sum(width > np.power(2.0, -s)) > 0:
                choice = self.A_s[np.argmax(width > np.power(2.0, -s))]

                regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = regret
                self.accurate_so_far[j] = correct_labels

                self.Z = np.append(self.Z, self.create_zi(self.X[j, :], choice).reshape(-1, 1), axis=1)
                self.past_reward = np.append(self.past_reward, self.r[self.y[j], choice])

            else:  # np.sum(width > 1/np.sqrt(self.T)) == 0
                choice = self.A_s[np.argmax(ucb)]

                regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = regret
                self.accurate_so_far[j] = correct_labels
