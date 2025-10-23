import numpy as np
import matplotlib.pyplot as plt
import cost_function


def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w_tmp  = np.zeros_like(X_train[0])
    b_tmp  = 0.
    alph = 0.1
    iters = 10000
    num_iters = []
    J = []
    w,b = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters, num_iters, J)
    print(w,b)
    plt.plot(num_iters, J)
    plt.show()

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    n = x.shape[1]

    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        z = np.dot(w,x[i]) + b
        f_w_b = cost_function.sigmoid(z)
        for j in range(n):
            dj_dw_i = (f_w_b - y[i])*x[i, j]
            dj_dw[j] += dj_dw_i
            
        dj_db_i = f_w_b - y[i]
        dj_db += dj_db_i


    return dj_dw/m, dj_db/m

def gradient_descent(x,y,w,b,alpha,iter, num_iters, J):
    
    for i in range(iter):
        dw, db = compute_gradient(x,y,w,b)
        tmp_w = w - alpha*(dw) 
        tmp_b = b - alpha*(db)

        w = tmp_w
        b = tmp_b
        num_iters.append(i)
        J.append(cost_function.cost_function(x,y,w,b))


    return w, b

if __name__ == "__main__":
    main()