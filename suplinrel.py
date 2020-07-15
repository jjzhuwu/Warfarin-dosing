import numpy as np
import LinRel

class SupLinRel(LinRel):

    def run(self):

        self.true_reward = 0
        self.regret = 0

        s=1

        for j in range(self.T):
            if self.Z.shape[1] < self.d:
                choice = self.A_s[np.random.randint(len(self.A_s))]
                self.true_reward += self.orig_reward[y[j], choice]
                self.regret += np.max(self.orig_reward[y[j], :])-self.orig_reward[y[j], choice]
                self.Z = np.append([self.Z, self.create_zi(x[j, :], choice).reshape(-1, 1)], axis=1)

            else:
                ucb, width = self.compute_ucb(j)

                while np.sum(width > np.power(2, -s)) == 0 and np.sum(width > 1/np.sqrt(self.T)) > 0:
                    self.Z = np.zeros((self.d+self.K-1, 0))
                    s += 1
                    A_s = A_s[ucb >= np.max(ucb)-2*power(2, -s)]
                    ucb, width = self.compute_ucb(j)

                if np.sum(width > np.power(2, -s)) > 0:
                    choice = self.A_s[np.argmax(width > np.power(2, -s))]
                    self.true_reward += self.orig_reward[y[j], choice]
                    self.regret += np.max(self.orig_reward[y[j], :])-self.orig_reward[y[j], choice]
                    self.Z = np.append([self.Z, self.create_zi(x[j, :], choice).reshape(-1, 1)], axis=1)

                else:  # np.sum(width > 1/np.sqrt(self.T)) == 0
                    choice = self.A_s[np.argmax(ucb)]
                    self.true_reward += self.orig_reward[y[j], choice]
                    self.regret += np.max(self.orig_reward[y[j], :])-self.orig_reward[y[j], choice]
