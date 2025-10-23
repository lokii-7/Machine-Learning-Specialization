import numpy as np

def z_normalization(x):
    m = x.shape[0]
    n = x.shape[1]
    std = np.std(x, axis=0)
    mean = np.mean(x, axis=0)
    z = np.zeros(x.shape)

    # for i in range(m):
    #     for j in range(n):
    #         z[i,j]= (x[i,j] - mean[j])/std[j]

    z = (x-mean)/std

    return z         