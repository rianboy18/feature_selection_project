# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
def select_from_model(data):
    X=data.drop(['SalePrice'],axis=1)
    y=data['SalePrice']
    rfc = RandomForestClassifier()
    sm = SelectFromModel(rfc)
    model = sm.fit(X,y)
    result=model.get_support()
    res=[]
    for i,j in zip(result,list(X)):
        if i==True:
            res.append(j)
    return res
c=select_from_model(data)
c


