from Preprocessing import *
from PolynomialRegression import *

x_train, x_test, y_train, y_test = preprocessing()
poly_error = polynomial_regression(x_train, x_test, y_train, y_test)
