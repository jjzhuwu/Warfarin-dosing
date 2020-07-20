import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from load_data import Load_Data
from linreg_bandit import LinReg_Bandit
from utils import mean_confidence_interval

alpha = 1
num_run = 20

X, y = Load_Data().extract()
threshold = np.array([21, 49])

reward = np.array([[0, -1, -1], \
                    [-1, 0, -1], \
                    [-1, -1, 0]])

regrets = np.zeros(num_run)
accuracy_scores = np.zeros(num_run)

for i in range(num_run):
    print("Processing Trial # %d out of %d." % (i+1, num_run))
    linreg = LinReg_Bandit(threshold, reward, alpha=alpha)
    linreg.data_load(X, y)
    linreg.run(initial_guess=35)
    regrets[i] = linreg.total_regret()
    accuracy_scores[i] = linreg.accuracy_score()

regrets_mean, regrets_width = mean_confidence_interval(regrets)
acc_mean, acc_width = mean_confidence_interval(accuracy_scores)

plt.scatter(np.zeros_like(regrets), regrets)
plt.xlim(-0.2, 0.2)
plt.fill_between(np.array([-0.1, 0.1]), np.full(2, regrets_mean-regrets_width), \
        np.full(2, regrets_mean+regrets_width), color='b', alpha=.1)
plt.hlines(regrets_mean, -0.1, 0.1)
plt.text(0.1, regrets_mean-regrets_width, np.round(regrets_mean-regrets_width, 2))
plt.text(0.1, regrets_mean, np.round(regrets_mean, 2))
plt.text(0.1, regrets_mean+regrets_width, np.round(regrets_mean+regrets_width, 2))

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title("Regret Confidence Interval with alpha = "+str(alpha))
plt.savefig("output/linreg_Regret_confidence_interval")
plt.clf()

plt.scatter(np.zeros_like(accuracy_scores), accuracy_scores)
plt.xlim(-0.2, 0.2)
plt.fill_between(np.array([-0.1, 0.1]), np.full(2, acc_mean-acc_width), \
        np.full(2, acc_mean+acc_width), color='b', alpha=.1)
plt.hlines(acc_mean, -0.1, 0.1)
plt.text(0.1, acc_mean-acc_width,  np.round(acc_mean-acc_width, 5))
plt.text(0.1, acc_mean, np.round(acc_mean, 5))
plt.text(0.1, acc_mean+acc_width,  np.round(acc_mean+acc_width, 5))

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title("Accuracy Confidence Interval with alpha = "+str(alpha))
plt.savefig("output/linreg_Accuracy_confidence_interval")
plt.clf()
