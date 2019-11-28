import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression_all(x_train, x_test, y_train, y_test):
    print(x_train[0])
    poly_features = PolynomialFeatures(degree=2)

    # transforms the existing features to higher degree features.
    x_train_poly = poly_features.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(x_train_poly, y_train)

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(x_test))

    error = metrics.mean_squared_error(y_test, prediction)
    print('Mean Square Error of Polynomial Regression with 7 features: ', error)

    return error
