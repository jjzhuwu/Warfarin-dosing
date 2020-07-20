import numpy as np
import pandas as pd

from load_data import Load_Data
from suplinrel import SupLinRel

from utils import dose_to_category

delta = 0.75

X, y = Load_Data().extract()
y = dose_to_category(y)

K = 3
reward = np.array([[0, -1, -1], \
                    [-1, 0, -1], \
                    [-1, -1, 0]])

SLR = SupLinRel(K, reward, delta=delta)
SLR.data_load(X, y)
SLR.run()

print("The regret is %f." % SLR.total_regret())
print("The accuracy score is %f." % (SLR.accuracy_score()))

SLR.plot_hist()
SLR.plot_recent_accuracy(N=500)
