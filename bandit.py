import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    model_name = ""

    def data_load(self, X, y, shuffle=True):
        pass

    def run(self):
        pass

    def accuracy_score(self):
        return self.accurate_so_far[-1] / self.T

    def total_regret(self):
        return self.regret_history[-1]

    def plot_hist(self):
        plt.plot(self.regret_history)
        plt.title("Regret History over Time")
        plt.savefig("output/"+self.model_name+"_Regret_history")
        plt.clf()

        plt.plot(self.accurate_so_far/np.array(range(1, self.T+1)))
        plt.title("Accuracy Score History")
        plt.savefig("output/"+self.model_name+"_Accuracy_history")
        plt.clf()

    def plot_recent_accuracy(self, N=500):
        recent_acc = np.zeros(self.T-N+1)
        recent_acc[0] = self.accurate_so_far[N-1]
        recent_acc[1:self.T-N+1] = self.accurate_so_far[N:self.T]-self.accurate_so_far[0:self.T-N]

        plt.plot(recent_acc/N)
        plt.text(self.T-N-200, recent_acc[-1]/N, str(recent_acc[-1]/N))
        plt.title("Running Accuracy of Latest %d Trials over Time" % N)
        plt.savefig("output/"+self.model_name+"_Running_accuracy")
        plt.clf()
