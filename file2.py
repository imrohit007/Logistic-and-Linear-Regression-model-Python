# A python program to implement simple logistic regression model

#Importing the libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import show, scatter, contourf

#Generating a random dataset
np.random.seed(0)
x = np.random.randn(100,2)  #100 rows and 2 columns
y = np.where(x[:,0] + x[:,1] > 0, 1, 0)  #if x1 + x2 > 0, y = 1 else y = 0

#Initializing the model
model = LogisticRegression()

#Fitting the model on the data
model.fit(x,y)

#plotting the decision boundary
x_min, x_max = x[:,0].min() - 0.5, x[:,0].max() + 0.5
y_min, y_max = x[:,1].min() - 0.5, x[:,1].max() + 0.5
h = 0.02 #step size in the mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = model.predict(np.c_[xx.ravel(), yy.ravel()])

#plotting the decision boundary
z = z.reshape(xx.shape)
contourf(xx, yy, z, cmap='rainbow', alpha=0.5)
scatter(x[:,0], x[:,1], c=y, edgecolors='k')
show()
