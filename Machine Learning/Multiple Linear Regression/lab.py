import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import gradient_descent


def main():
    x_train, y_train = load_data()
    x_train.reshape(-1, 1)
    plt.scatter(x_train, y_train, marker='x', c='r')
    plt.title('Profits vs. Population per city')
    plt.ylabel('Profit in $10,000')
    plt.xlabel('Population of City in 10,000s')
    w_init = np.zeros(x_train.shape[0])
    b_init = 0.0
    alpha = 0.001
    J = []
    num_iters = []
    w,b = gradient_descent.gradient_descent(x_train, y_train, w_init, b_init, alpha, 10000, J, num_iters)
    y_predit = np.dot(w,x_train) + b
    plt.plot(x_train, y_predit)
    plt.show()

def load_data():
    data = np.loadtxt('data.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y

if __name__ == "__main__":
    main()


