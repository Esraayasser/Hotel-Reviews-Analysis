from sklearn import linear_model
from sklearn import metrics
import numpy as np


def Multiple_Linear_Regression1(x_train, x_test, y_train, y_test):
    x_train = x_train[:, [2, 5, 7]]
    x_test = x_test[:, [2, 5, 7]]
    cls = linear_model.LinearRegression()

    cls.fit(x_train, y_train)
    prediction = cls.predict(x_test)
    test_error = metrics.mean_squared_error(np.asarray(y_test), prediction)

    print('Mean Square Train Error for multiple linear regression on the top features', test_error)

    # test_error = metrics.mean_squared_error(np.asarray(y_test)[0], prediction[0])
    # print('Mean Square Test Error for multiple linear regression on the top features', train_error)
    # return train_error, test_error
    return test_error


def MultipleLinearRegressionPositiveNegative(x_train, x_test, y_train, y_test):
    x_train = x_train[:, [5, 7]]
    x_test = x_test[:, [5, 7]]

    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    test_error = metrics.mean_squared_error(np.asarray(y_test), prediction)
    print('Mean Square Error for Multiple Linear Regression on Positive and Negative Word Count', test_error)
    return test_error
