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

    def __init__(self, K, reward, delta=0.1):

        """
        Parameters:
            K: number of actions
            reward: matrix of size K*K, reward[i][j] gives the deterministic
                reward when the true label is i and the prediction is j
        """
        self.K = K
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
        self.past_reward = np.zeros(0)

        if shuffle:
            indexes = np.random.permutation(T)
            self.X = self.X[indexes, :]
            self.y = self.y[indexes, :]

    def normalize_zi(self, zi):

        """
        Normalize the data so that zi has L2(Euclidean) norm at most 1.
        Assuming zi = [x_i, 1_{i=1}, ..., 1_{i=K-1}]
        """
        normed_zi = zi/self.X_max/np.sqrt(self.d+self.K-1)
        return normed_zi

    def step(self, j):

        x_j = self.X[j, :].T
        y_j = self.y[j, :]

        if j < self.d:
            pass
        else:
            D, U = np.linalg.eig(self.Z @ self.Z.T)
            U = U.T
            mask_geq_1 = D >= 1
            D[D < 0.5] = 0.5
            D_inv_masked = np.diag(1/D * mask_geq_1)

            """
            W contains the vectors z_i
            U_tilde contains the vectors u_i tilde
            V_tilde contains the vectorss v_i tilde
            """
            W = np.zeros((self.d+self.K-1, self.K))
            for i in range(self.K):
                zi = np.zeros(self.d+self.K-1)
                zi[0:x_j.shape[0]] = x_j
                if i > 0:
                    zi[x_j.shape[0]+i-1]=1
                W[:, i] = normalize_zi(zi)

            UW = U @ W

            U_tilde = (UW.T * mask_geq_1).T
            V_tilde = UW - U_tilde

            """
            A contains the transpose of a_i

            """

            A = U_tilde.T @ D_inv_masked @ U @ Z

            width = np.sqrt(np.sum(A**2, axis=1)*np.log(2*self.T*self.K/self.delta)) + np.sqrt(np.sum(A**2, axis=0))
            ucb = (self.past_reward @  A.T).reshape(-1) + width

            choice = np.argmax(ucb)

        return 0

    def train(self):
        for j in range(self.T):
            self.step(j)
