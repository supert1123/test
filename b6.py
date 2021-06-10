import numpy as np
from funtion import *


#np.std() -- hàm tính độ lệch chuẩn

[X,y] = Load('data.txt')
[X,mu,s] = Norm(X)
[theta, J_his] = Gad(X,y,0.2,400)
input = np.array([1,1650,3])
input = (input-mu)/s
input[0] = 1
predict = predict(input,theta)
print('%.2f'%(predict))