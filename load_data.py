import pandas as pd

class Load_Data:

    yname = 'Therapeutic Dose of Warfarin'

    def __init__(self, filename = "data/warfarin.csv", target_clense = True):

        self.data = pd.read_csv(filename)
        if target_clense:
            self.data = self.data.loc[self.data[self.yname].notnull()]
        self.clense()

    def clense(self):
        """
            Data Clensing:
            Age to Age_in_decade, imputing nan by mode of age group
            Height: imputing nan by the mean
            Weight: imputing nan by the mean
            Race: get dummies
            Enzyme inducer: imputing nan by 0
            Amiodarone: impting nan by 0

            VKORC1 and CYP2C9: using data in consensus, treating unknown as another type
        """
        self.data['Age_in_decade'] = self.data['Age'].fillna(self.data['Age'].mode()[0]).astype('str').map(lambda s: s[0]).astype('int')
        self.data['Height (cm)'] = self.data['Height (cm)'].fillna(self.data['Height (cm)'].mean())
        self.data['Weight (kg)'] = self.data['Weight (kg)'].fillna(self.data['Weight (kg)'].mean())

        race_dummies = pd.get_dummies(self.data['Race'], prefix='Race')
        self.race_categories = list(race_dummies.columns)
        self.data = pd.concat([self.data, race_dummies], axis=1)

        enzyme_inducer_parts = ['Carbamazepine (Tegretol)', 'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']
        self.data[enzyme_inducer_parts] = self.data[enzyme_inducer_parts].fillna(0)
        self.data['enzyme_inducer'] = self.data[enzyme_inducer_parts].max(axis=1)

        self.data['Amiodarone (Cordarone)'] = self.data['Amiodarone (Cordarone)'].fillna(0)

        self.data['VKORC1 -1639 consensus'] = self.data['VKORC1 -1639 consensus'].fillna('Unknown')
        self.data['CYP2C9 consensus'] = self.data['CYP2C9 consensus'].fillna('Unknown')

        genes_dummies = pd.concat([pd.get_dummies(self.data['VKORC1 -1639 consensus'], prefix='VKORC1'),\
         pd.get_dummies(self.data['CYP2C9 consensus'], prefix='CYP2C9')], axis=1)
        self.genes_categories = list(genes_dummies.columns)
        self.data = pd.concat([self.data, genes_dummies], axis=1)

        self.data = pd.concat([self.data, pd.get_dummies(self.data['VKORC1 -1639 consensus']),\
         pd.get_dummies(self.data['CYP2C9 consensus'])], axis=1)

    def extract(self, genotype = False):

        self.choosen_columns = ['Age_in_decade', 'Height (cm)', 'Weight (kg)'] + self.race_categories + ['enzyme_inducer', 'Amiodarone (Cordarone)']

        if genotype:
            self.choosen_columns += self.genes_categories

        return self.data[self.choosen_columns], self.data[self.yname]
