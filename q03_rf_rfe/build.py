# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(data):
    X = data.drop(['SalePrice'],axis=1)
    y = data['SalePrice']
    model = RandomForestClassifier()
    rfe = RFE(model)
    rfe.fit(X,y)
    res=[]
    for i,j in zip((rfe.ranking_),(list(X))):
        if i==1:
            res.append(j)
    return res


c=rf_rfe(data)
c


