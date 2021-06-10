import numpy as np
from numpy.core.records import array
#np.array(object, dtype= None , ndmin= 0 )
A = [[1,2,3],[3,4,5],[5,6,7]]
C = [[3,5,7],[2,4,6]]
B = np.squeeze(np.array(A))
D = np.squeeze(np.array(C))
print(B)
#print(D)
#A = np.eye(3) tạo ma trận có hàng và cột = nhau đường chéo chính = 1
#E = B.dot(D)
##print(B@D)
#print('a[0,2]', B[0,2])
#print('a[1:2]', B[1,2])
#print('a[0,:]', B[0,:])
#print('a[:,2', B[:,2])
E = np.transpose(A) # đảo ma trận
print(np.size(E,0))
print(np.min(A,0)) # tìm hàng có giá trị nhỏ nhẩt
print(np.max(A,1)) # tìm cột có giá trị lớn nhất
print(E)