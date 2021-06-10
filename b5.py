import numpy as np
from funtion import *
import matplotlib.pyplot as plt


dat = np.loadtxt('data.txt',dtype=int, delimiter=',')
# tách lấy X
X = np.copy(dat)
X[:,1] = X[:,0]
#thêm bias 1
X[:,0] = 1
#tách y
y = dat[:,1]

[theta, J_his] = Gad(X,y)
predict = predict(X,theta)*10000
plt.figure(1)
plt.plot(X[:,1],y,'rx')
plt.plot(X[:,1],predict/10000,'-b')
plt.show