from sklearn import linear_model
from sklearn import metrics
import numpy as np


def Multiple_Linear_Regression1(x_train, x_test, y_train, y_test):
    x_train = x_train[:, [1, 4, 6]]
    x_test = x_test[:, [1, 4, 6]]
    cls = linear_model.LinearRegression()

    cls.fit(x_train, y_train)
    prediction = cls.predict(x_test)
    train_error = metrics.mean_squared_error(np.asarray(y_test), prediction)

    print('Co-efficient of linear regression', cls.coef_)
    print('Intercept of linear regression model', cls.intercept_)
    print('Mean Square Train Error for multiple linear regression on the top features', train_error)

    test_error = metrics.mean_squared_error(np.asarray(y_test)[0], prediction[0])
    print('Mean Square Test Error for multiple linear regression on the top features', train_error)
    return train_error, test_error

