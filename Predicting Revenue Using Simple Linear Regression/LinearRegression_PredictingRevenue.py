--------------------------------------------------------------
### Project: REVENUE PREDICTION - SIMPLE LINEAR REGRESSION ###
--------------------------------------------------------------
---------------------------------
### STEP 1: PROBLEM STATEMENT ###
---------------------------------
'''
PROBLEM STATEMENT
You own an ice cream business and you would like to create a model that could predict the daily revenue in dollars based on the outside air temperature (degC). You decide that a Linear Regression model might be a good candidate to solve this problem.
Data set:

Independant variable X: Outside Air Temperature
Dependant variable Y: Overall daily revenue generated in dollars
'''
--------------------------------
### STEP 2: LIBRARIES IMPORT ###
--------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

------------------------------
### STEP 3: IMPORT DATASET ###
------------------------------
IceCream = pd.read_csv("IceCreamData.csv")
IceCream.head()
IceCream.tail()
IceCream.describe()
IceCream.info()

---------------------------------
### STEP 4: VISUALIZE DATASET ###
---------------------------------
sns.jointplot(x='Temperature', y='Revenue', data = IceCream)
sns.pairplot(IceCream)
sns.lmplot(x='Temperature', y='Revenue', data=IceCream)

---------------------------------------------------
### STEP 5: CREATE TESTING AND TRAINING DATASET ###
---------------------------------------------------
y = IceCream['Revenue']
X = IceCream[['Temperature']]

from sklearn.model_selection import train_test_split

#splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

-------------------------------
### STEP 6: TRAIN THE MODEL ###
-------------------------------
X_train.shape

from sklearn.linear_model import LinearRegression

#instatiation of an object out of our class
#when "fit_intercept = True" - asking the model to obtain intercept which is value of 'm' and 'b'
#when "fit_intercept = False" - model will obtain only the 'm' value; 'b' will be zero by default
regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)

print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)

------------------------------
### STEP 7: TEST THE MODEL ###
------------------------------
y_predict = regressor.predict( X_test)
y_predict
y_test

#VISUALIZE TRAIN SET RESULTS
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand(Training dataset)')

#VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Hours')
plt.title('Revenue Generated vs. Hours @Ice Cream Stand(Test dataset)')

y_predict = regressor.predict(30)
y_predict
