from Preprocessing import *
from PolynomialRegression import *
from LinearRegression import *
from MultipleLinearRegression import *
import SVM
import Testing

# Initialization
models = []  # list of models
models_names = []  # list of models names
models_training_time = []  # list of models training time

# Pre processing
X_train = [], y_train = []  # list of training samples
X_test = [], y_test = []  # list of testing samples

# Training models
model, time = SVM.svm_one_versus_one_training(X_train, y_train)
models.append(model)
models_names.append('svm-one-versus-one')
models_training_time.append(time)

model, time = SVM.svm_one_versus_rest_training(X_train, y_train)
models.append(model)
models_names.append('svm-one-versus-all')
models_training_time.append(time)

# Testing Models
models_accuracies, models_testing_time = Testing.model_testing(models, X_test, y_test)

# Printing Models outputs
for i in range(len(models)):
    print("Model name is {} - accuracy = {} and model time = {}".
          format(models_names[i], models_accuracies*[i], models_testing_time[i]))

# Regression Milestone1
'''
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
'''
