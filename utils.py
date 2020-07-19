import numpy as np
import scipy.stats

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    w = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, w
