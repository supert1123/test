
import numpy as np
from funtion import *


dat = np.loadtxt('data.txt', delimiter=',')
print(dat[0:5,:])
y= dat[:,2]
X = np.zeros((np.size(y),np.size(dat,1)))
X[:,0]=1
X[:,1:]=dat[:,0:2]
theta = np.array([0,1,2])
print(Cost(X,y,theta),Cost_Vec(X,y,theta))