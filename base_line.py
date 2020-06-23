import numpy as np
import pandas as pd

import load_data, fixed_dose

X, y = load_data.Load_Data().extract()

model = fixed_dose.Fixed_Dose()
model.fit(X, y)
ypred = model.predict(X)

print("The accuracy score is:", np.sum(ypred == y)/y.shape[0])
