import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0])

# Target values
y = np.array([3.0, 5.0, 7.0, 9.0])

def compute_gradient(x,y,w,b):
     dj_dw = 0
     dj_db = 0
     n = x.shape[0]
     for i in range(n):
          dj_dwi = ((w*x[i] + b - y[i])*x[i])/n
          dj_dbi = ((w*x[i] + b - y[i]))/n
          dj_dw += dj_dwi
          dj_db += dj_dbi

     return dj_dw, dj_db



def gradient_descent(x, y, alpha, num_iters):
     w, b = 0, 0
     for _ in range(num_iters):
          temp_w = w - alpha*compute_gradient(x,y,w,b)[0]
          temp_b = b - alpha*compute_gradient(x,y,w,b)[1]
          w = temp_w
          b = temp_b
     return w,b
w,b = gradient_descent(x, y, 0.01, 10000)
print(f"w: {w}, b: {b}")
