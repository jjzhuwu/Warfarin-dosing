import numpy as np
import pandas as pd

import load_data, baseline

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


X, y = load_data.Load_Data().extract(genotype=True)

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
