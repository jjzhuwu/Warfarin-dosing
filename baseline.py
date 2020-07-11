import numpy as np

class Baseline:

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

class Fixed_Dose(Baseline):

    def predict(self, X):
        return np.array([35 for i in range(X.shape[0])])

class Clinical_Dosing_Alg(Baseline):

    coef = np.array([[4.0376, -0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443,\
        0, 1.2799, -0.5695]])

    def predict(self, X):
        """
        Requires the columns to be same as load_data.Load_Data.choosen_columns
        """
        ones_X = np.concatenate((np.ones((X.shape[0], 1)), X.iloc[:, 0:9]), axis=1)
        return np.square(ones_X@self.coef.T)

class Pharma_Dosing_Alg(Baseline):

    def predict(self, X):

        sqaured_dosage = 5.6044-0.2614*X['Age_in_decade']\
            +0.0087*X['Height (cm)'] +0.0128*X['Weight (kg)'] \
            -0.8677*X['VKORC1_A/G'] -1.6974*X['VKORC1_A/A'] \
            -0.4854*X['VKORC1_Unknown']-0.5211*X['CYP2C9_*1/*2'] \
            -0.9357*X['CYP2C9_*1/*3']-1.0616*X['CYP2C9_*2/*2'] \
            -1.9206*X['CYP2C9_*2/*3']-2.3312*X['CYP2C9_*3/*3'] \
            -0.2188*X['CYP2C9_Unknown']-0.1092*X['Race_Asian'] \
            -0.2760*X['Race_Black or African American'] \
            -0.1032*X['Race_Unknown']+1.1816*X['enzyme_inducer']\
            -0.5503*X['Amiodarone (Cordarone)']

        return np.square(sqaured_dosage)
