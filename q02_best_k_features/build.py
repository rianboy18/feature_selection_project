# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


def percentile_k_features(data,k=20):
    X=data.drop(['SalePrice'],axis=1)
    y=data['SalePrice']
    
    SP=SelectPercentile(f_regression,percentile=k)
    model=SP.fit_transform(X,y)
    return model
c=percentile_k_features(data,k=20)
c




