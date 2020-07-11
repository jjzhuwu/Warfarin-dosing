import numpy as np

class LinRel:

    def normalize_reward(self, reward):

        """
        Normalize reward such that rewards are in [0, 1]
        """
        reward_max = np.max(reward)
        reward_min = np.min(reward)
        normed_reward = (reward-reward_min)/reward_max
        return normed_reward

    def __init__(self, k, reward, delta=0.1):

        """
        Parameters:
            k: number of actions
            reward: matrix of size k*k, reward[i][j] gives the deterministic
                reward when the true label is i and the prediction is j
        """
        self.k = k
        self.orig_reward = reward
        self.r = self.normalize_reward(reward)

        self.delta = delta

    def data_load(self, X, y, shuffle=True):

        self.X = X
        self.X_max = np.max(X, axis=0)
        self.y = y

        self.T = X.shape[0]
        self.d = X.shape[1]
        self.Z = np.zeros((self.d, 0))

        if shuffle:
            indexes = np.random.permutation(T)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes, :]

    def normalize_zi(self, zi):

        """
        Normalize the data so that zi has L2(Euclidean) norm at most 1.
        Assuming zi = [x_i 1_{i=1}, ..., 1_{i=k-1}]
        """
        zi = zi/self.X_max/np.sqrt(self.d+self.k-1)
        return zi

    def step(self, i, y):

        if i < self.d:
            pass
        else:
            D, U = np.linalg.eig(self.Z @ self.Z.T)
            U = U.T
            pass
        return 0

    def train(self):
        for i in range(self.T):
            self.step(self.X[i, :], self.y[i])
