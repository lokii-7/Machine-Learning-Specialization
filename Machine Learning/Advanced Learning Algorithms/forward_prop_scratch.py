import numpy as np


class Layer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

    def train(self, x):
        n = x.shape[0]
        w_layer = np.random.rand(self.units, x.shape[1])
        b_layer = np.zeros(self.units)
        a = np.zeros((n, self.units))

        for i in range(n): 
            for j in range(self.units):
                a_tmp = np.dot(w_layer[j], x[i]) + b_layer[j]
                a[i,j] = 1/(1+np.exp(-1*a_tmp))

        return a 

x_train = np.array([
    [100, 3, 45],
    [50, 2, 10],
])         
layer_1 = Layer(units= 2, activation="sigmoid")

output = layer_1.train(x_train)