import numpy as np
import pandas as pd

import baseline
from load_data import Load_Data
from utils import dose_to_category

X, y = Load_Data().extract(genotype=True)

model1 = baseline.Fixed_Dose()
model1.fit(X, y)
ypred = model1.predict(X)

print("The accuracy score of fixed dose is: %f." % np.round(np.mean(dose_to_category(ypred) == dose_to_category(y)), 6))

model2 = baseline.Clinical_Dosing_Alg()
model2.fit(X, y)
ypred = model2.predict(X)

print("The accuracy score of clinical dosing algorithm is: %f." % np.round(np.mean(dose_to_category(ypred) == dose_to_category(y)), 6))

model3 = baseline.Pharma_Dosing_Alg()
model3.fit(X, y)
ypred = model3.predict(X)

print("The accuracy score of Pharmacogenetic dosing algorithm is: %f." % np.round(np.mean(dose_to_category(ypred) == dose_to_category(y)), 6))
