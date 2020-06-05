# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:37:54 2019

@author: Dell
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv("merged2.csv")

X = dataset.iloc[:,0:31].values
y = dataset.iloc[:,31:32].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform' , activation = 'relu', input_dim = 31 ))


#Adding the second hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform' , activation = 'relu' ))


#Adding the third hidden layer
classifier.add(Dense(output_dim = 16, init = 'uniform' , activation = 'relu' ))


#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid' ))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
#fitting ANN to the Training set
classifier.fit(X_train, y_train, batch_size =1000 , nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = classifier.predict(sc.transform(np.array([[0.00E+00,0.004954129,-0.00135893,0.004767455,-0.008526795,-0.00118725,0.004382603,-0.000651758,-0.03157736,-0.007815173,0.00280153,0.00610745,0.005736843,0.005886622,0.00395724,-0.001407764,	0.001572102,	-0.001740023,	-0.001308009,	0.003346061,	-0.000695411	,0.000771683,	-0.001459152	,0.002673313,	-0.001022762	,0.000871385,	-0.003662355	,0.005480237,	-0.004953712,	0.003154866,	-0.002783455]
])))
if (y_pred>0.5):
    y_pred = True
else:
      y_pred = False
  
    
y_pred2 = classifier.predict(sc.transform(np.array([[2.93E-03,	-0.01978657,	0.005272564,	-0.003470697,	0.004499115,	-0.02361103,	0.004458822,	0.0128046,	0.02860562,	0.009632573,	0.0045349	,-0.004647456,	0.0162496	,-0.01543104,	0.01038449,	-0.0140401,	0.005387217,	-0.004955489,	0.009316035,	-0.003598712,	0.006795108,	-0.001712944,	0.007826921,	0.000537507,	0.009287003,	0.000920057,	-0.001097824	,-0.00259589	,0.01336877	,-0.00136384	,0.003096842]])))
if (y_pred2>0.5):
    y_pred2 = True
    
else:
    y_pred2 = False
    
    
y_pred3 = classifier.predict(sc.transform(np.array([[1.56E-02,	-0.05745137	,-0.01929993,	-0.01470184	,0.01333202	,-0.04396094,	-0.04732147	,-0.04605642	,-0.006040871	,-0.04604396	,-0.03166394	,-0.0208198	,0.02659251	,-0.05626581	,0.0307616	,-0.03393319,	0.03228329	,0.009362843,	0.01875691	,-0.001492471	,0.03549647	,-0.0688064	,0.01738056	,-0.04961761,	0.01526522	,-0.06686546,	0.02153684	,-0.02995655,	0.0701599,	-0.03773249	,0.02832786]])))      
if (y_pred3>0.5):
    y_pred3 = True
    
else:
    y_pred3 = False
  

y_pred4 = classifier.predict(sc.transform(np.array([[1.07E-02	,0.02266581	,-0.05025517	,0.04628153	,-0.05691658,	0.008091589	,0.02481791,	0.020448	,-0.04353936	,0.002702039	,0.02185528	,-0.03014893	,-0.01361173	,0.003174966	,-0.01348101	,-0.03064758	,0.01970412	,-0.02633141	,-0.001082268	,-0.02214634	,0.01889488	,0.007505111	,0.0216178	,0.007586624,	0.02598811	,0.01304164	,-0.01933327	,-0.003960259	,-0.003659457	,-0.006797105	,-0.0148313]])))
if (y_pred4>0.5):
    y_pred4 = True
    
else:
    y_pred4 = False
   

y_pred5 = classifier.predict(sc.transform(np.array([[2.64E-02	,0.05141491,	0.1371299,	0.01410442,	0.05277888,	0.08098627,	0.0765454,	-0.03402597,	-0.03920335,	0.009691,	0.03192964,	0.03976386,	0.01606074,	0.01855531,	-0.01870973,	0.08450627,	-0.0469705,	0.00365792,	-0.02952722,	0.02933582,	-0.0469683,	0.07505307,	-0.04142286,	0.024223,	-0.07352217,	0.06647334,	-0.06438492,	0.07537909,	-0.1402242,	0.08180247,	-0.06571731]])))
if (y_pred5>0.5):
    y_pred5 = True
    
else:
    y_pred5 = False
   
    
    
y_pred6 = classifier.predict(sc.transform(np.array([[1.86E-02	,-0.02970519	,-0.146157	,-0.01000619	,-0.06614485	,-0.1572252	,-0.04276229	,0.1335134	,-0.3211988	,0.07301298	,-0.01995921	,-0.1028809	,0.001871698	,-0.03145802	,0.006327968	,-0.118453	,0.07570876	,-0.0677254	,0.04361693	,-0.08076902	,0.07341506	,-0.05298574	,0.05783384	,-0.04011805	,0.05608254	,-0.06115766	,0.0548408	,-0.1656903	,0.09615924	,-0.1994943	,0.03413431]])))
if (y_pred6>0.5):
    y_pred6 = True
   
else:
    y_pred6 = False
  


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




















