import numpy as np
import matplotlib.pyplot as plt


A = np.loadtxt('HS.txt', delimiter = ',')
X = A[:,0],A[:,1]
y = A[:,2]

true_x = []
true_y = []
false_x = []
false_y = []
for item in  A:
    if item[2] == 1. :
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_x.append(item[0])
        false_y.append(item[1])


plt.scatter(true_x,true_y, marker='o', c='y')
plt.scatter(false_x,false_y, marker='s', c='g')
#plt.show()
def sigmoid(z) :
    return 1/(1+np.exp(-z))
def phanchia(p): # dua vao` xac xuat chia P
    if p >= 0.5:
        return 1
    else:
        return 0


def predict(feature, weight):
    z = np.dot(feature, weight)
    return sigmoid(z)


def cost_function(feature,labels,weight):
    n = len(labels)
    pre = predict(feature, weight)
    cost_c1 = -labels*np.log(pre)
    cost_c2 = -(1-labels)*np.log(1-pre)
    cost = cost_c1 + cost_c2
    return cost.sum()/n



def update_weigh(feature,labels,weight,learning_rate):
    n = len(labels)


    pre = predict(feature,weight)
    gd = np.dot(feature.T ,(predict-labels))
    gd = gd/n
    gd = gd*learning_rate
    weight = weight-gd
    return weight


def training(feature,labels,weight,learning_rate,iter):
    cost_his = []
    for i in range(iter):
        weight = update_weigh(feature,labels,weight,learning_rate)
        cost = cost_function(feature,labels,weight)
        cost_his.append(cost)
    return weight, cost_his


weight, cost_his = training(X,y,0.5,0.05,50)
print(weight)
print(cost_his)





    