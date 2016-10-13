#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2

def plotErrorall(w):
  for iter in range (0,max_iter):
    y = sps.expit(np.dot(X,w))
    e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
    e_all.append(e)
    grad_e = np.mean(np.multiply((y - t), X.T), axis=1)
    w_old = w
    w = w - eta*grad_e
    
def reset(w, e_all):
  w = np.array([0.1, 0, 0])
  del e_all[:]

max_iter = 500
tol = 0.00001
data = np.genfromtxt('data.txt')
np.random.shuffle(data)
X = data[:,0:3]
t = data[:,3]
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]
w = np.array([0.1, 0, 0])
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

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(['0.5','0.3','0.1','0.05','0.01'])
plt.show()
