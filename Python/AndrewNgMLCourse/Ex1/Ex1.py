# Machine Learning course by AndrewNg on Coursera

# import all libraries required here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

# 1.  Print 5 by 5 Identity matrix
id5 = np.identity(5)
print(id5)

# 2.  Plot the data
data = np.loadtxt('ex1data1.txt', delimiter=',')

X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]


#data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit']) # read from dataset
#print(data.head()) # view first few rows of the data
#print(data.describe())

#X = data.iloc[:,0] # read first column
#y = data.iloc[:,1] # read second column
m = len(y) # number of training example
plt.scatter(X[:,1], y)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# 3.  Cost function
# Linear regression equation: h(x[i]) = theta0 + theta1*x[i]
# Linear regression Cost function, J(theta0, theta1) = (1/2m) * sum[h(x[i]) -
# y[i]]^2, where 1 <= i <= m
# partial-derivative w.r.t.  theta(j): (1/m)*sum[h(x[i]) - y[i]]*x[i], where 1
# <= i <= m

# Gradient descent:
# repeat until convergence {
#   theta(j) := theta(j) + alpha * partial-derivative w.r.t.  theta(j) of
#   J(theta0, theta1), for j = 0 and 1 simultaneously
# }
# Simplified Gradient descent algo:
# repeat until convergence {
#   theta(j) := theta(j) + (alpha/m) * sum[h(x[i]) - y[i]]*x[i], for j = 0 and
#   1 simultaneously and 1 <= i <= m
# }

# Assign initially theta0 and theta1 = 0, alpha = 0.01
theta = np.zeros([2,1])
alpha = 0.01

# add function to compute cost
def computeCost(X,y,theta, m):
    temp = X.dot(theta) - y
    return np.sum(np.power(temp, 2)) / (2 * m)

# Compute cost at current values
J = computeCost(X, y, theta, m)
print(J)

# add functino to calculate gradient descent
def gradientDescent(X, y, theta, alpha, m, iterations = 1500):
    J_history = np.zeros(iterations)

    for iter in np.arange(iterations):
        h = X.dot(theta)
        theta = theta - (alpha/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X,y,theta, m)
    return (theta, J_history)

# Compute gradient descent or theta for minimized cost J
theta, Cost_J = gradientDescent(X,y,theta, alpha, m)

print('theta: ', theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')
plt.show()

# Plot the actual data and linear regression plot
xx = np.arange(5,23)
yy = theta[0] + theta[1]*xx

# Plot gradient descent
plt.scatter(X[:,1],y,s=30,c='r',marker='+',linewidths =1)
plt.plot(xx,yy,label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4);
plt.show()

# predict profit for a city with population of 35000 and 70000
print(theta.T.dot([1,3.5])*10000)
print(theta.T.dot([1,7])*10000)