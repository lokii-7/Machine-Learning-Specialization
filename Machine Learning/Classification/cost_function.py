import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss_function(f_w_b, y_i):
    return (y_i * (math.log(f_w_b))) + ((1 - y_i)*(math.log(1- f_w_b)))

def cost_function(x, y, w, b):
    m = x.shape[0]
    J = 0.0

    for i in range(m):
        z = np.dot(w,x[i]) + b
        f_w_b = sigmoid(z)

        loss = loss_function(f_w_b, y[i])
        J += loss

    J = (-1*J)/m

    return J



def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1]) 
    w_tmp = np.array([1,1])
    b_tmp = -3                                          
    a = cost_function(X_train, y_train, w_tmp, b_tmp)

    print(a)




if __name__ == "__main__":
    main()