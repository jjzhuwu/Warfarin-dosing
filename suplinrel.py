import numpy as np
from linrel import LinRel
import matplotlib.pyplot as plt

class SupLinRel(LinRel):

    def run(self):

        self.true_reward = 0
        self.regret = 0

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

                self.true_reward += self.orig_reward[self.y[j], choice]
                self.regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = self.regret
                self.accurate_so_far[j] = correct_labels

                self.Z = np.append(self.Z, self.create_zi(self.X[j, :], choice).reshape(-1, 1), axis=1)
                self.past_reward = np.append(self.past_reward, self.r[self.y[j], choice])

            elif np.sum(width > np.power(2.0, -s)) > 0:
                choice = self.A_s[np.argmax(width > np.power(2.0, -s))]

                self.true_reward += self.orig_reward[self.y[j], choice]
                self.regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = self.regret
                self.accurate_so_far[j] = correct_labels

                self.Z = np.append(self.Z, self.create_zi(self.X[j, :], choice).reshape(-1, 1), axis=1)
                self.past_reward = np.append(self.past_reward, self.r[self.y[j], choice])

            else:  # np.sum(width > 1/np.sqrt(self.T)) == 0
                choice = self.A_s[np.argmax(ucb)]
                self.true_reward += self.orig_reward[self.y[j], choice]
                self.regret += np.max(self.orig_reward[self.y[j], :])-self.orig_reward[self.y[j], choice]
                correct_labels += self.y[j] == choice
                self.regret_history[j] = self.regret
                self.accurate_so_far[j] = correct_labels

    def plot_hist(self):
        plt.plot(self.regret_history)
        plt.title("Regret History over Time")
        plt.savefig("Regret_history")
        plt.clf()

        plt.plot(self.accurate_so_far/np.array(range(1, self.T+1)))
        plt.title("Accuracy Score History")
        plt.savefig("Accuracy_history")
        plt.clf()

    def plot_recent_accuracy(self, N=1000):
        recent_acc = np.zeros(self.T-N+1)
        recent_acc[0] = self.accurate_so_far[N-1]
        recent_acc[1:self.T-N+1] = self.accurate_so_far[N:self.T]-self.accurate_so_far[0:self.T-N]

        plt.plot(recent_acc/N)
        plt.text(self.T-N-200, recent_acc[-1]/N, str(recent_acc[-1]/N))
        plt.title("Running Accuracy of Latest %d Trials over Time" % N)
        plt.savefig("Running_accuracy")
        plt.clf()
