
    https://colab.research.google.com/drive/1G9YPdjMy5mnorZAGo0yIt5aNybwr8M1E
        

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np

data=pd.read_csv('train (1).csv')
df=data.isnull().sum()
data.drop(['MiscFeature'],axis=1,inplace=True)
strin=data.select_dtypes(include=['object'])
ar=strin.columns.values
strin=strin.fillna(0)

dict={'HouseStyle':{'1Story':1, '2Story':2, '1.5Fin':1.5, '1.5Unf':1.25 ,'SLvl':1.1, 'SFoyer':1.1, '2.5Fin':2.5, '2.5Unf':2.2}}
strin.replace(dict,inplace=True)
encoded_data=pd.get_dummies(strin,columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
       'Condition1', 'Condition2', 'BldgType', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
       'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
       'SaleType', 'SaleCondition'])

encoded_data=encoded_data.astype('int64')

data=data.select_dtypes(exclude='object')
y=data.SalePrice
scaler = MinMaxScaler()
scaled_data= pd.DataFrame(scaler.fit_transform(data),columns=data.columns)

training_data=scaled_data.join(encoded_data)
x=training_data.drop(columns='SalePrice')
x=x.fillna(0)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 4)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 100, num = 8)]
max_depth.append(None)
min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model= LGBMRegressor(random_state=1)

model.fit(x,y)
inp=pd.read_csv('cvsf2.csv')
pred=model.predict(inp)
out=pd.DataFrame(pred)
ind=pd.read_csv('test.csv')
i=pd.DataFrame(ind['Id'])
out=i.join(out)
out.columns=['Id','SalePrice']
out.to_csv('final.csv',index=False)
