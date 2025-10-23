import gradient_descent
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.arange(0, 20, 1)
    y = 1 + x**2

    X = x.reshape(-1, 1)
    J = []
    num_iters = []
    w = np.zeros(X.shape[1])
    model_w, model_b = gradient_descent.gradient_descent(X, y, w=w, b=0, alpha=1e-2, num_iter=1000, J=J, num_iters=num_iters)

    plt.scatter(x, y, c="r", label= "actual value")
    plt.plot(X, np.dot(X, model_w) + model_b, c="b", label="predicted value")
    plt.show()

if __name__ == "__main__":
    main()