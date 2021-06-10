import numpy as np
import matplotlib.pyplot as plt


#nhập xuất data của file = numpy
#np.loadtxt(fname,dtype,delimiter)
X = np.loadtxt('univariate.txt',delimiter = ',')
print(X[0:5,:])
theta = np.loadtxt('univariate_theta.txt', delimiter=',')
#lưu cột y lại
y = np.copy(X[:,-1])
#chuyển cột đầu tiên sang cột 2
X[:,1] = X[:,0]
#đổi cột đầu thành số 1
X[:,0] = 1
#tính lợi nhuận (Đv = 10000$)
pre = X.dot(theta)
predict = pre*10000
#in cặp dân số-lợi nhuận đàu tiên
print('%d người : %.2f' %(X[0,1]*1000,predict[0]))
#lưu file
np.savetxt('predict.txt', predict, fmt='%.6f')
#plot giá trị thực tế k lấy cột bias)
plt.plot(X[:,1:],y,'rx')
plt.plot(X[:,1:],predict/10000,'-b')
#dự đoán plot
plt.show()



A = np.loadtxt('multivariate.txt',delimiter = ',')
theta1= np.loadtxt('multivariate_theta.txt',delimiter=',')
y1 = np.copy(A[:,-1])
A[:,1] = A[:,0]
A[:,0] = 1
pre1 = A@theta1
predict1 = pre1*10000
print('%.2f feet-vuông %d phòng ngủ : %.1f$' %(A[0,1], A[0,2], predict1[0]))
np.savetxt('predict1.txt',predict1, fmt='%.6f')
plt.plot(A[:,1:],y1,'rx')
plt.plot(A[:,1:],predict1/10000,'-b')
plt.show()
