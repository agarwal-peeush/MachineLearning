# Machine Learning course by AndrewNg on Coursera

# import all libraries required here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Print 5 by 5 Identity matrix

id5 = np.identity(5)
print(id5)

# 2. Plot the data
data = pd.read_csv('ex1data1.txt', header=None) # read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
print(data.head()) # view first few rows of the data

plt.scatter(X, y)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()