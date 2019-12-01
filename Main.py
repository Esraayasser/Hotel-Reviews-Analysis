from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *
from MultipleLinearRegression import *

x_train, x_test, y_train, y_test, X, Y = preprocessing()

linearAS_train_error, linearAS_test_error, modelAS = Linear_Regression_averageScore(x_train, x_test, y_train, y_test)
PlotLinearModel(X[:, 1], Y, modelAS, 'Average Score', 'Reviewer Score')

linearPRC_train_Error, LinearPRC_test_Error, modelPRC = Linear_Regression_PositiveReviewCount(x_train, x_test, y_train, y_test)
PlotLinearModel(X[:, 6], Y, modelPRC, 'Positive Review Word Count', 'Reviewer Score')

linearNRC_train_error, linearNRC_test_error, modelNRC = LinearRegressionNegativeReviewCount(x_train, x_test, y_train, y_test)
PlotLinearModel(X[:, 4], Y, modelNRC, 'Negative Review Word Count', 'Reviewer Score')

#multipleLR1_train_Error, multipleLR1_test_Error = Multiple_Linear_Regression1(x_train, x_test, y_train, y_test)
multipleLR1_test_Error = Multiple_Linear_Regression1(x_train, x_test, y_train, y_test)
multipleLR_PN_error = MultipleLinearRegressionPositiveNegative(x_train, x_test, y_train, y_test)

poly1_error_train, poly1_error_test = polynomial_regression_all(x_train, x_test, y_train, y_test)
poly2_error_train, poly2_error_test = polynomial_regression_top_three(x_train, x_test, y_train, y_test)
