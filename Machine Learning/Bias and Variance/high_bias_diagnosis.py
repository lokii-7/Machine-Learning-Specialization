import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , PolynomialFeatures
from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

def accuracy(y_predict, y):
    correct = 0
    m = len(y_predict)
    for i in range(m):
        if y_predict[i]>=0.5:
            y_predict[i] = 1
        else:
            y_predict[i] = 0

    for i in range(m):
        if y_predict[i] == y[i]:
            correct += 1

    return correct/m


def plot_curve(x_train, x_cv, y_true_train, y_true_cv):
    m = len(y_true_train)
    m_graph = []
    J_t = []
    J_cv = []
    for i in range(1,m,20):
        x = x_train[:i]
        y = (y_true_train[:i])

        if len(np.unique(y))<2:
            continue

        plot_model = LogisticRegression()
        plot_model.fit(x,y)
        y_train = plot_model.predict_proba(x)
        y_cv = plot_model.predict_proba(x_cv)
        m_graph.append(i)
        j_train = log_loss(y_pred=y_train, y_true= y)
        j_cv = log_loss(y_pred=y_cv, y_true= y_true_cv)
        J_t.append(j_train)
        J_cv.append(j_cv)
    
    return m_graph, J_t, J_cv

x, y = sklearn.datasets.load_breast_cancer()["data"], sklearn.datasets.load_breast_cancer()["target"]

x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.4, random_state= 1)

x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size= 0.5, random_state= 1)

del x_, y_


x_train_feature_selection = x_train[:, list(range(3))]

x_cv_feature_selection = x_cv[:, list(range(3))]


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_feature_selection)
model = LogisticRegression()
x_cv_scaled = scaler.transform(x_cv_feature_selection)

model.fit(x_train_scaled, y_train)

y_hat = model.predict_proba(x_train_scaled)

j_train = log_loss(y_pred=y_hat, y_true=y_train)

print(f"Training Error: {j_train}")

y_hat_cv = model.predict_proba(x_cv_scaled)

j_cv = log_loss(y_pred=y_hat_cv, y_true=y_cv)

print(f"Validation Error: {j_cv}")

m_values, j_train_vals, j_cv_vals = plot_curve(x_train_scaled, x_cv_scaled, y_train, y_cv)

plt.figure(figsize=(10, 6))
plt.plot(m_values, j_train_vals, label="Training Error")
plt.plot(m_values, j_cv_vals, label="Cross-Validation Error", linestyle="--")
plt.title("Learning Curve (Underfit Model)")
plt.xlabel("Training Set Size (m)")
plt.ylabel("Cost (J)")
plt.legend()
plt.grid(True)
plt.show()