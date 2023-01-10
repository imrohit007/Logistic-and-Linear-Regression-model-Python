#python program to implement a simple linear regression model

#Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

#Create a random dataset
x = np.array([1,2,3,4,5]).reshape((-1,1))
y = np.array([2,4,6,8,10])

#Create a linear regression model
model = LinearRegression()

#Train the model
model.fit(x,y)

#Print the model parameters
print("Intercept: ", model.intercept_)
print("Slope: ", model.coef_)
print("R-squared: ", model.score(x,y))

#Predict the output for a given input
print("Predicted output: ", model.predict([[6]]))
