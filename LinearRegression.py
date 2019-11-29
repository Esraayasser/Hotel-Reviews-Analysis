import numpy as np
from sklearn import linear_model
from sklearn import metrics


def Linear_Regression_averageScore(x_train, x_test, y_train, y_test):
    model = linear_model.LinearRegression()

    x_train = x_train[:, 1]

    x_train = np.expand_dims(x_train, axis=1)
   # y_train = np.expand_dims(y_train, axis=1)

    model.fit(x_train, y_train)

    prediction = model.predict(x_train)
    train_error = metrics.mean_squared_error(y_train, prediction)
    print('Mean Square Error of Linear Regression with Average Score feature on train set: ', train_error)

    x_test = x_test[:, 1]

    x_test = np.expand_dims(x_test, axis=1)
    #y_test = np.expand_dims(y_test, axis=1)

    prediction_test = model.predict(x_test)
    test_error = metrics.mean_squared_error(y_test, prediction_test)
    print('Mean Square Error of Linear Regression with Average Score feature on test set: ', test_error)

    return train_error, test_error


def Linear_Regression_PositiveReviewCount(x_train, x_test, y_train, y_test):
    model = linear_model.LinearRegression()

    x_train = x_train[:, 6]

    x_train = np.expand_dims(x_train, axis=1)

    model.fit(x_train, y_train)

    prediction = model.predict(x_train)
    train_error = metrics.mean_squared_error(y_train, prediction)
    print('Mean Square Error of Linear Regression with Review_Total_Positive_Word_Counts feature on train set: '
          , train_error)

    x_test = x_test[:, 6]

    x_test = np.expand_dims(x_test, axis=1)

    prediction_test = model.predict(x_test)
    test_error = metrics.mean_squared_error(y_test, prediction_test)
    print('Mean Square Error of Linear Regression with Review_Total_Positive_Word_Counts feature on test set: '
          , test_error)

    return train_error, test_error

