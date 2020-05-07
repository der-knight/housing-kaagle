
    https://colab.research.google.com/drive/1G9YPdjMy5mnorZAGo0yIt5aNybwr8M1E
        
#import pandas sklearn and numpy
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
#read training data
data=pd.read_csv('train (1).csv')
#get an idea of null values to drop null columns
df=data.isnull().sum()
data.drop(['MiscFeature'],axis=1,inplace=True)
#separate the obejct data type from float data type for encoding
strin=data.select_dtypes(include=['object'])
#get the column names 
ar=strin.columns.values
#The NAN values in the table are representative of no assests in home eg no fireplace
strin=strin.fillna(0)

# creating a dictionary to encode the number of stories
dict={'HouseStyle':{'1Story':1, '2Story':2, '1.5Fin':1.5, '1.5Unf':1.25 ,'SLvl':1.1, 'SFoyer':1.1, '2.5Fin':2.5, '2.5Unf':2.2}}
strin.replace(dict,inplace=True)
#encode the object type data using dummies one hot encoding
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
#get the floating point data
data=data.select_dtypes(exclude='object')
#get the prediction datasets
y=data.SalePrice
#scaler to scale from 0-1 although it does not improve accuracy in case of RF
scaler = MinMaxScaler()
scaled_data= pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
#join encoded data with floating point data
training_data=scaled_data.join(encoded_data)
#remove the target data from training data 
x=training_data.drop(columns='SalePrice')

x=x.fillna(0)
#grid for randomised search
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
#model 
model= RandomForestRegressor()
#model with parameters randomly searched for best identification
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x,y)
#import the testing data in encoded form using the same encoding
inp=pd.read_csv('cvsf2.csv')
#predict 
pred=rf_random.predict(inp)

out=pd.DataFrame(pred)
#import original tetsing data for the id merging 
ind=pd.read_csv('test.csv')
i=pd.DataFrame(ind['Id'])
out=i.join(out)
out.columns=['Id','SalePrice']
out.to_csv('final.csv',index=False)sv')
i=pd.DataFrame(ind['Id'])
#get column id and join
out=i.join(out)
out.columns=['Id','SalePrice']
out.to_csv('final.csv',index=False)
