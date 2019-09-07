# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 10:03:56 2019

@author: HIMANSHU R SINGH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("data.csv")

#seprating the dataset
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#no missing data 

#traing and testing
from sklearn.model_selection import train_test_split 
X_train , X_test, y_train ,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# avoiding dummy variable trp



#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test) 
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# because the library will take care of it

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting the test set results

y_pred = regressor.predict(X_test)






x= [[63,1,1,145,233,1,2,150,0,2.3,3,0,6]]
y_new =regressor.predict(x)
if y_new > 1:
    print("1")
elif y_new <0.3:
    print("o")
else:
    print(y_new)





#building an optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr =np.ones((297,1)).astype(int),  values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()
#
X_opt = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[1,2,3,4,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[2,3,4,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[2,3,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[2,3,6,7,8,9,10,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[2,3,7,8,9,10,12,13]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()