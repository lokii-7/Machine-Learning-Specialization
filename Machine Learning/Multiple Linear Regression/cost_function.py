import numpy as np

def main():

    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

    cost = cost_function(X_train, y_train, w_init, b_init)
    print(f'Cost at optimal w : {cost}')


def cost_function(x,y, w, b=0):
    J = 0
    m = x.shape[0]
    for i in range(m):
        error_squared = (np.dot(w,x[i]) + b - y[i])**2
        J = J + error_squared
    J = J/(2*m)
    return J


if __name__ == "__main__":
    main()

