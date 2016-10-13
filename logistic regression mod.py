#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

def plotErrorall(w):
  for iter in range (0,max_iter):
    # Compute output using current w on all data X.
    y = sps.expit(np.dot(X,w))
    # e is the error, negative log-likelihood (Eqn 4.90)
    e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
    # Add this error to the end of error vector.
    e_all.append(e)
    # Gradient of the error, using Eqn 4.91
    grad_e = np.mean(np.multiply((y - t), X.T), axis=1)
    # Update w, *subtracting* a step in the error derivative since we're minimizing
    w_old = w
    w = w - eta*grad_e
    
def reset(w, e_all):
  w = np.array([0.1, 0, 0])
  del e_all[:]

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001
# Load data.
data = np.genfromtxt('data.txt')
# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]
# Initialize w.
w = np.array([0.1, 0, 0])
# Error values over all iterations.
e_all = []

for x in xrange(1,6):
  if x==1:
    eta = 0.5
    plotErrorall(w)
    plt.plot(e_all)
  if x==2:
    eta = 0.3
    reset(w, e_all)
    plotErrorall(w)
    plt.plot(e_all)
  if x==3:
    eta = 0.1
    reset(w, e_all)
    plotErrorall(w)
    plt.plot(e_all)
  if x==4:
    eta = 0.05
    reset(w, e_all)
    plotErrorall(w)
    plt.plot(e_all)
  if x==5:
    eta = 0.01
    reset(w, e_all)
    plotErrorall(w)
    plt.plot(e_all)

# Plot error over iterations
# plt.figure()
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.show()
