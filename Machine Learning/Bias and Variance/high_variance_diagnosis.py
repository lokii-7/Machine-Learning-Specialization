import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from keras.layers import Dense
from keras import Sequential
from keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt

x, y = sklearn.datasets.load_breast_cancer()["data"], sklearn.datasets.load_breast_cancer()["target"]

x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state= 1)

x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size= 0.5, random_state= 1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_cv_scaled = scaler.transform(x_cv)

del x_, y_

nn = Sequential([
    Dense(units=25, activation="relu"),
    Dense(units=15, activation="relu"),
    Dense(units=5, activation="relu"),
    Dense(units=1, activation="sigmoid")
])

m = len(y_train)
j_train_list = []
j_cv_list = []
m_values = []
for i in range(1, m, 10):
    nn.compile(loss=BinaryCrossentropy())
    x = x_train_scaled[:i]
    y = y_train[:i]
    nn.fit(x, y, epochs=100)
    if len(np.unique(y))<2:
        continue


    y_train_predict = nn.predict(x)

    y_cv_predict = nn.predict(x_cv_scaled)

    j_train = log_loss(y_pred= y_train_predict, y_true=y)
    j_cv = log_loss(y_pred = y_cv_predict, y_true= y_cv)
    j_train_list.append(j_train)
    j_cv_list.append(j_cv)
    m_values.append(i)



plt.figure(figsize=(10, 6))
plt.plot(m_values, j_train_list, label="Training Error")
plt.plot(m_values, j_cv_list, label="Cross-Validation Error", linestyle="--")
plt.title("Learning Curve (Underfit Model)")
plt.xlabel("Training Set Size (m)")
plt.ylabel("Cost (J)")
plt.legend()
plt.grid(True)
plt.show()