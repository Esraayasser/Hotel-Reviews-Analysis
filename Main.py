from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *
from MultipleLinearRegression import *

x_train, x_test, y_train, y_test = preprocessing()
linear1_train_error, linear1_test_error = Linear_Regression_averageScore(x_train, x_test, y_train, y_test)
linearPRC_train_Error, LinearPRC_test_Error = Linear_Regression_PositiveReviewCount(x_train, x_test, y_train, y_test)
multipleLR1_train_Error, multipleLR1_test_Error = Multiple_Linear_Regression1(x_train, x_test, y_train, y_test)
poly1_error_train, poly1_error_test = polynomial_regression_all(x_train, x_test, y_train, y_test)
poly2_error_train, poly2_error_test = polynomial_regression_top_three(x_train, x_test, y_train, y_test)
