from funtion import *

import numpy as np
[X,y] = Load('data.txt')
[theta] = NormEq(X,y)
inp = np.array([1,1650,3])
predict = predict(inp,theta)

print('%.2f$'%(predict))