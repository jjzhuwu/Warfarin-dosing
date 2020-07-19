import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from load_data import Load_Data
from suplinrel import SupLinRel
from utils import dose_to_category, mean_confidence_interval

X, y = Load_Data().extract()
y = dose_to_category(y)

K = 3
reward = np.array([[0, -1, -1], \
                    [-1, 0, -1], \
                    [-1, -1, 0]])

num_run = 20

regrets = np.zeros(num_run)
accuracy_scores = np.zeros(num_run)

for i in range(num_run):
    print("Processing Trial # %d out of %d." % (i+1, num_run))
    SLR = SupLinRel(K, reward, delta=0.5)
    SLR.data_load(X, y)
    SLR.run()
    regrets[i] = SLR.total_regret()
    accuracy_scores[i] = SLR.accuracy_score()

regrets_mean, regrets_width = mean_confidence_interval(regrets)
acc_mean, acc_width = mean_confidence_interval(accuracy_scores)

fig, ax = plt.subplots()
ax.scatter(np.zeros_like(regrets), regrets)
ax.set_xlim(-0.2, 0.2)
ax.fill_between(np.array([-0.1, 0.1]), np.full(2, regrets_mean-regrets_width), \
        np.full(2, regrets_mean+regrets_width), color='b', alpha=.1)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title("Regret Confidence Interval")
plt.savefig("SupLinRel_Regret_confidence_interval")
plt.clf()

fig, ax = plt.subplots()
ax.scatter(np.zeros_like(accuracy_scores), accuracy_scores)
ax.set_xlim(-0.2, 0.2)
ax.fill_between(np.array([-0.1, 0.1]), np.full(2, acc_mean-acc_width), \
        np.full(2, acc_mean+acc_width), color='b', alpha=.1)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.title("Accuracy Confidence Interval")
plt.savefig("SupLinRel_Accuracy_confidence_interval")
plt.clf()
