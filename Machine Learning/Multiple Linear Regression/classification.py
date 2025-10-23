import gradient_descent
import numpy as np
import matplotlib.pyplot as plt


# Training data features (X)
X_train = np.array([[ 2,  2],
                    [ 3,  3],
                    [ 4,  1],
                    [-1, -1],
                    [-2, -3],
                    [-1, -2],
                    [ 2,  0],
                    [-1,  1]])

# Training data labels (y)
# (Using a 1D array is common)
y_train = np.array([1, 1, 1, 0, 0, 0, 1, 0])

# Test data (for your own verification)
X_test = np.array([[ 3,  1],
                   [-2, -2],
                   [ 0,  0],
                   [ 5, -1]])

alpha = 0.03
J = []
num_iters = []
X = gradient_descent.z_normalization(X_train)

initial_w = np.zeros_like(X_train.shape)
initial_b = 0
w,b = gradient_descent.gradient_descent(X, y_train, initial_w, initial_b, alpha, 10000, J, num_iters)
J_array = np.array(J)
iter_array = np.array(num_iters)

plt.plot(iter_array, J_array)
plt.show()

print(w,b)

model_output = np.dot(w, X_train[0]) + b
print(model_output)

