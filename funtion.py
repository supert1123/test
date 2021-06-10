import numpy as np




def predict(X,theta):
    return X.dot(theta)


# cách 1 dài dòng dễ phân tích, dễ hiểu
def Cost(X,y,theta):
    pre = predict(X,theta)
    sqr_error = (pre - y)**2
    sum_error = np.sum(sqr_error)
    m = np.size(y)
    J = (1/(2*m))*sum_error
    return J
# Cách 2 Ngắn gọn, súc tích
def Cost_Vec(X,y,theta):
    error = predict(X,theta) - y
    m = np.size(y)
    J = (1/(2*m))*np.transpose(error).dot(error)
    return J

def Gad(X,y,alpha=0.02,iter=5000):
    #số lượng theta = số lượng cột X
    theta = np.zeros(np.size(X,1))
    J_hist = np.zeros((iter,2))
    #kích thước của training set
    m = np.size(y)
    #ma trận đảo của cột X
    X_ = np.transpose(X)
    #biến kiểm tra tiến độ
    pre_cost = Cost(X,y,theta)
    for i in range(0,iter):
        error = predict(X,theta) - y
        #theta mới
        theta = theta - (alpha/m)*(X_.dot(error))
        #tính J hiện tại
        costp = Cost(X,y,theta)
        #so sánh với J của vòng lập trước(so sánh 15 chữ số thập phân)
        if np.round(costp,15) == np.round(pre_cost,15):
            #in ra vòng lập hiện tại
            print('Ngưỡng của I = %d; J = %.6f' %(i,costp))
            #thêm tất cả index còn lại sau khi break
            J_hist[i:,0] = range(i,iter)
            # Giá trị của i sau khi break vẫn như cũ
            J_hist[i:,1] = costp
            break
    

        costp = pre_cost
        J_hist[i,0] = i
        J_hist[i,0] = costp
    yield theta
    yield J_hist






def Norm(X):  #( Chia đọ lệch chuẩn)
    n = np.copy(X)
    n[0,0] = 100
    #tính std cho từng feature
    s = np.std(n,0, dtype= np.float64)
    mu  = np.mean(n,0)
    n = (n-mu)/s
    #gán lại X0 = 1
    n[:,0] = 1
    yield n 
    yield mu
    yield s


def Load(path):
    try: 
        dat = np.loadtxt(path,dtype= int, delimiter = ',')
        X = np.zeros((np.size(dat,0),np.size(dat,1)))
        X[:,0] = 1
        X[:,1:] = dat[:,:-1]
        y = dat[:,-1]
        yield X
        yield y 
    except:
        return 0




def NormEq(X,y):
    return np.linalg.pinv(X.T.dot(X))@(X.T.dot(y))