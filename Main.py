from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *

x_train, x_test, y_train, y_test = preprocessing()
poly_error = polynomial_regression_all(x_train, x_test, y_train, y_test)
train_error , test_error = Linear_Regression_averageScore(x_train, x_test, y_train, y_test)
lin1, lin2 = Linear_Regression_averageScore(x_train, x_test, y_train, y_test)