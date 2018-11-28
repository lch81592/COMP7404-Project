# Mui Kei Lam Alvin COMP7404 Nov 2018

import math
import numpy as np
import pandas as pd
from numpy import loadtxt, where
from pandas import DataFrame
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# create training data set
# formats the input data into independant variables and dependant variable
train_datafile = pd.read_csv("Train_30min.csv", header=0)
X_train = train_datafile[["ROC5","Vol.Change","ATR","ROC10","OBV","MFI","TRIX","MACD","ADX","TRIX.Sig"]]
X_train = np.array(X_train)
Y1 = np.ravel(train_datafile["Ret30Min"])
Y_train = np.zeros(Y1.shape[0])
for i in range(Y1.shape[0]):
        if (Y1[i] == "UP"):
                Y_train[i] = 1
        elif (Y1[i] == "DOWN"):
                Y_train[i] = 2
        else:
                Y_train[i] = 0

# create testing data set
# formats the input data into independant variables and dependant variable
test_datafile = pd.read_csv("Test_30min.csv", header=0)
X_test = test_datafile[["ROC5","Vol.Change","ATR","ROC10","OBV","MFI","TRIX","MACD","ADX","TRIX.Sig"]]
X_test = np.array(X_test)
Y2 = np.ravel(test_datafile["Ret30Min"])
Y_test = np.zeros(Y2.shape[0])
for i in range(Y2.shape[0]):
        if (Y2[i] == "UP"):
                Y_test[i] = 1
        elif (Y2[i] == "DOWN"):
                Y_test[i] = 2
        else:
                Y_test[i] = 0

# make an instance of the scikit-learn model and train the model
clf = LogisticRegression(solver='lbfgs')
clffit = clf.fit(X_train,Y_train)
score = clf.score(X_test, Y_test)
print('Accuracy = ', score)

# make predictions on entire test data
Y_predict = clffit.predict(X_test)
print('Y_prediction on entire test data = ', Y_predict)

# plot data on a graph
pos = where(Y_train == 1)
neg = where(Y_train == 2)
scatter(X_train[pos, 0], X_train[pos, 1], marker='o', c='r')
scatter(X_train[neg, 0], X_train[neg, 1], marker='x', c='b')
xlabel('ROC5')
ylabel('Vol.Change')
legend(['UP', 'DOWN'])
show()

# The sigmoid function adjusts the cost function hypotheses
def sigmoid(z):
	gz = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return gz

# The hypothesis is the linear combination of all the known factors x[i] and
# their current estimated coefficients theta[i] 
# This hypothesis is used to calculate each instance of the cost function
def hypothesis(theta, x):
	z = 0
	for i in range(len(theta)):
		z += x[i]*theta[i]
	return sigmoid(z)

# For each member of the dataset, the result (Y) determines which variation of the cost function is used
# The Y = 0 cost function punishes high probability estimations, and the Y = 1 cost function punishes low scores
# The punishment makes the change in the gradient of ThetaCurrent - Average(cost(Dataset)) greater
def cost(X,Y,theta,m):
	sumErrors = 0
	for i in range(m):
		hi = hypothesis(theta,X[i])
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		else:
			error = 0
		sumErrors += error
	const = -1/m
	J = const * sumErrors
	return J

# This function creates the gradient component for each Theta value 
# The gradient is the partial derivative by Theta of the current value of theta minus 
# a "learning speed factor alpha" times the average of all the cost functions for that theta
# For each Theta there is a cost function calculated for each member of the dataset
def cost_derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in range(m):
		xi = X[i]
		xij = xi[j]
		hi = hypothesis(theta,xi)
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

# For each theta, the partial differential 
# The gradient, or vector from the current point in Theta-space (each theta value is its own dimension) to the more accurate point, 
# is the vector with each dimensional component being the partial differential for each theta value
def gradient_descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in range(len(theta)):
		CFDerivative = cost_derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

# The high level function for the LR algorithm which, for a number of steps (num_iter) finds gradients which take 
# the Theta values (coefficients of known factors) from an estimation closer (new_theta) to their "optimum estimation" which is the
# set of values best representing the system in a linear combination model
def logistic_regression(X,Y,alpha,theta,num_iter):
	m = len(Y)
	for x in range(num_iter):
		new_theta = gradient_descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			# the cost function is used to present the final hypothesis of the model in the same form for each gradient-step iteration
			print('theta = ', theta)
			print('cost = ', cost(X,Y,theta,m))

# These are the initial guesses for theta as well as the learning rate of the algorithm
initial_theta = [0,0]
alpha = 0.1
print('Initial theta = ', initial_theta)
print('Learning rate alpha = ', alpha)
iterations = 1000
print('Confusion matrix without normalization:')
print(confusion_matrix(Y_test,Y_predict))
print('Training set:')
logistic_regression(X_train,Y_train,alpha,initial_theta,iterations)
print('Testing set:')
logistic_regression(X_test,Y_test,alpha,initial_theta,iterations)
