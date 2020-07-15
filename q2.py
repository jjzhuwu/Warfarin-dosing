import numpy as np
import pandas as pd

from load_data import Load_Data
from suplinrel import SupLinRel

def dose_to_category(y):

    """
    Convert dose from continuous to categorical variables. Label 0 for low dosage, label 1
    for medium dosage, and label 2 for high dosage.
    """
    if len(y.shape) > 1:
        y = y.reshape(-1)
    low_bound = 21
    high_bound = 49
    y_cat = np.zeros((y.shape[0], 3))
    y_cat[:,0] = y < low_bound
    y_cat[:,1] = np.logical_and(y >= low_bound, y <= high_bound)
    y_cat[:,2] = y > high_bound
    return np.argmax(y_cat, axis=1)

X, y = Load_Data().extract()
y = dose_to_category(y)

K = 3
reward = np.array([[0, -1, -1], \
                    [-1, 0, -1], \
                    [-1, -1, 0]])

SLR = SupLinRel(K, reward, delta=0.5)
SLR.data_load(X, y)
SLR.run()

print("The reward is %f." % SLR.true_reward)
print("The regret is %f." % SLR.regret)
print("The accuracy score is %f." % (SLR.accurate_so_far[-1] / SLR.T))

SLR.plot_hist()
SLR.plot_recent_accuracy(N=500)
