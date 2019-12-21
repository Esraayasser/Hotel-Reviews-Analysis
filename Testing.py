from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
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
