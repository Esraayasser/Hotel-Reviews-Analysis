from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
import time
import numpy as np


def model_testing(models, x_test, y_test):
    models_accuracies = []
    models_testing_time = []

    for i in range(len(models)):
        start_time = time.time()
        predictions = models[i].predict(x_test)
        end_time = time.time()
        models_testing_time.append(abs(start_time - end_time))
        accuracy = np.mean(predictions == y_test)
        models_accuracies.append(accuracy)

    return models_accuracies, models_testing_time


def regression_testing(models, x_test, y_test):
    models_mean_square_error = []
    models_testing_time = []
    models_r2_score = []

    for i in range(len(models)):
        start_time = time.time()

        prediction_test = models[i].predict(x_test)
        test_error = metrics.mean_squared_error(y_test, prediction_test)
        r2_score = metrics.r2_score(y_test, prediction_test)

        end_time = time.time()

        models_r2_score.append(r2_score)
        models_mean_square_error.append(test_error)
        models_testing_time.append(abs(start_time - end_time))

        return models_testing_time, models_r2_score, models_mean_square_error
