import numpy as np
import matplotlib.pyplot as plt

def computing_cost(x, y, w,b):
    J = 0
    m = len(x)
    for i in range(0, m):
        error_squared = ((w*x[i])+b - y[i])**2
        J += (error_squared)/(2*m)

    return J

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])








