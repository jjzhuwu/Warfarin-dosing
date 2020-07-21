import numpy as np

class LinRel:

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

    def normalize_reward(self, reward):

        """
        Normalize reward such that rewards are in [0, 1]
        """
        reward_max = np.max(reward)
        reward_min = np.min(reward)
        normed_reward = (reward-reward_min)/(reward_max-reward_min)
        return normed_reward

    def create_zi(self, x_j, i):

        """
        Create and normalize z_i so that normed_z_i has L2(Euclidean) norm at most 1.
        Assuming z_i = [x_j, 1_{i=1}, ..., 1_{i=K-1}]
        """
        z_i = np.zeros(self.d+self.K-1)
        z_i[0:x_j.shape[0]] = x_j
        if i > 0:
            z_i[x_j.shape[0]+i-1]=1
        normed_z_i = z_i/self.Z_max/np.sqrt(self.d+self.K-1)
        return normed_z_i

    def compute_ucb(self, j):

        """
            A_s is a subset of {0, 1, ..., K-1}
        """
        x_j = self.X[j, :].T

        if self.Z.shape[1] < self.d:
            return np.array([]), np.array([])
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
            W = np.zeros((self.d+self.K-1, self.A_s.shape[0]))
            for i in range(self.A_s.shape[0]):
                W[:, i] = self.create_zi(x_j, self.A_s[i])

            UW = U @ W

            U_tilde = (UW.T * mask_geq_1).T
            V_tilde = UW - U_tilde

            """
            A contains the transpose of a_i
            """

            A = U_tilde.T @ D_inv_masked @ U @ self.Z

            width = np.sqrt(np.sum(A**2, axis=1)*np.log(2*self.T*self.K/self.delta)) + np.sqrt(np.sum(V_tilde**2, axis=0))
            ucb = (self.past_reward @  A.T).reshape(-1) + width

        return ucb, width
