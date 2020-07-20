import numpy as np
import pandas as pd

from load_data import Load_Data
from linreg_bandit import LinReg_Bandit

alpha = 1.0

X, y = Load_Data().extract()

threshold = np.array([21, 49])

reward = np.array([[0, -1, -1], \
                    [-1, 0, -1], \
                    [-1, -1, 0]])

linreg = LinReg_Bandit(threshold, reward, alpha=alpha)
linreg.data_load(X, y)
linreg.run(initial_guess=35)

print("The regret is %f." % linreg.total_regret())
print("The accuracy score is %f." % (linreg.accuracy_score()))

linreg.plot_hist()
linreg.plot_recent_accuracy(N=500)
