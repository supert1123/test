import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

A = np.loadtxt('univariate.txt', delimiter=',')

X = A[:, 0]
y = A[:, 1]
def predict(val,weight,bias):
    return weight*val + bias

def cost_function(X,y, weight,bias):
    n=len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i]-(weight*X[i]+bias))**2
    return sum_error/n
def update_weight(X,y, weight,bias, learning_rate):
    n=len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*X[i]*(y[i]-(X[i]*weight+bias))
        bias_temp +=  -2*(y[i]-(X[i]*weight+bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate
    return weight, bias
def train(X,y,weight,bias,learning_rate,iter):
    for i in range(iter):
        cost_his = []
        weight,bias = update_weight(X,y, weight,bias, learning_rate)
        cost = cost_function(X,y, weight,bias)
        cost_his.append(cost)
    return weight,bias,cost
weight,bias,cost = train(X,y,0.5,0.05,0.005,50)
print('kết quả: ')
print(weight)
print(bias)
print(cost)
print('giá trị dự đoán', predict(19,weight,bias))
    
