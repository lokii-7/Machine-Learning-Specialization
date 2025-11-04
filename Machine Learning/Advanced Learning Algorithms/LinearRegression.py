import numpy as np
import sklearn.model_selection
from sklearn.preprocessing import PolynomialFeatures , StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = np.loadtxt("data.txt", delimiter=",")

x,y = data[:, 0], data[:, 1]

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

x_train, x_, y_train, y_ = sklearn.model_selection.train_test_split(x, y, test_size= 0.40, random_state=1)

x_cv, x_test, y_cv, y_test = sklearn.model_selection.train_test_split(x_, y_, test_size= 0.20, random_state= 1)

del x_, y_

scalers = []
models = []
train_mse = []
cv_mse = []


for degree in range(1, 11):
    poly = PolynomialFeatures(degree, include_bias=False)
    x_train_mapped = poly.fit_transform(x_train)

    scalar = StandardScaler()
    x_train_mapped_scaled = scalar.fit_transform(x_train_mapped)
    scalers.append(scalar)

    model = LinearRegression()
    model.fit(x_train_mapped_scaled, y_train)
    models.append(model)



    yhat_train = model.predict(x_train_mapped_scaled)
    tr_mse = mean_squared_error(y_train, yhat_train)
    train_mse.append(tr_mse)

    x_cv_mapped = poly.transform(x_cv)
    x_cv_mapped_scaled = scalar.transform(x_cv_mapped)

    y_cv_predict = model.predict(x_cv_mapped_scaled)

    cvmse = mean_squared_error(y_cv, y_cv_predict)

    cv_mse.append(cvmse)

degrees=range(1,11)

plt.plot(degrees, train_mse)
plt.plot(degrees, cv_mse)

plt.show()


degree = np.argmin(cv_mse) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")
