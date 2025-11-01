import numpy as np

def back_prop(x,y,w,b):
    alpha = 0.001
    for i in range(100):
        z = np.dot(w,x) + b
        dl_dz = ((-1*np.exp(-z))/(np.exp(-z) + 1)) + (1-y)

        dl_dw = np.dot(dl_dz, x.T)/x.shape[1]

        dl_db = np.sum(dl_dz)/x.shape[1]


        temp_w = w - alpha*dl_dw
        temp_b = b - alpha*dl_db

        w = temp_w
        b = temp_b
    
    return w, b