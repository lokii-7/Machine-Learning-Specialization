import numpy as np
import matplotlib.pyplot as plt

# 1. Create some sample data
def main():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
    alpha = 0.0000000000001
    J = []
    num_iters = []
    X = z_normalization(X_train)

    initial_w = np.zeros_like(w_init)
    initial_b = 0
    w,b = gradient_descent(X, y_train, initial_w, initial_b, alpha, 10000, J, num_iters)
    J_array = np.array(J)
    iter_array = np.array(num_iters)

    plt.plot(iter_array, J_array)
    plt.show()

    print(w,b)


def cost_function(x,y, w, b=0):
    J = 0
    m = x.shape[0]
    for i in range(m):
        error_squared = (np.dot(w,x[i]) + b - y[i])**2
        J = J + error_squared
    J = J/(2*m)
    return J

def compute_gradient(x,y, w, b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for j in range(n):
        t = 0
        for i in range(m):
            t += (np.dot(w, x[i]) + b - y[i])*x[i,j]
        t = t/m
        dj_dw[j] = t

    for i in range(m):
        dj_db += (np.dot(w,x[i]) + b - y[i])
    
    dj_db /= m
    return dj_dw, dj_db



def gradient_descent(x, y, w, b, alpha, num_iter, J, num_iters):
    n = x.shape[1]
    for j in range(num_iter):
        dj = compute_gradient(x,y,w,b)
        for i in range(n):
            temp_w = w[i] - alpha*dj[0][i]
            w[i] = temp_w
        temp_b = b - alpha*dj[1]
        b = temp_b
        J.append(cost_function(x, y, w, b))
        num_iters.append(j)
        

    return w,b

def z_normalization(x):
    std = np.std(x, axis=0)
    mean = np.mean(x, axis=0)
    z = np.zeros(x.shape)

    # for i in range(m):
    #     for j in range(n):
    #         z[i,j]= (x[i,j] - mean[j])/std[j]

    z = (x-mean)/std

    return z         

if __name__ == "__main__":
    main()

