# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:16:00 2020

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset1 = pd.read_csv("C:/Users/Dell/Desktop/DNN/merged1.csv")
X1 = dataset1.iloc[:, 0:31].values
y1 = dataset1.iloc[:, 31:].values
                  
#splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
X1_train = sc1.fit_transform(X1_train)
X1_test = sc1.transform(X1_test)
y1_train = y1_train.reshape(589823,1)

#fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X1_train,y1_train.ravel())


# Visualising the Random Forest Regression results (higher resolution)
# Predicting the Test set results
y_pred1 = regressor.predict(X1_test)
# Passing one of the data row to check our algorithm's result
y_pred1 = regressor.predict(sc1.transform(np.array([[0.00E+00,0.004954129,-0.00135893,0.004767455,-0.008526795,-0.00118725,0.004382603,-0.000651758,-0.03157736,-0.007815173,0.00280153,0.00610745,0.005736843,0.005886622,0.00395724,-0.001407764,	0.001572102,	-0.001740023,	-0.001308009,	0.003346061,	-0.000695411	,0.000771683,	-0.001459152	,0.002673313,	-0.001022762	,0.000871385,	-0.003662355	,0.005480237,	-0.004953712,	0.003154866,	-0.002783455]
])))
int(np.ceil(y_pred1))

