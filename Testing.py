from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import time


def model_testing(models, x_test, y_test):
    models_accuracies = []
    models_testing_time = []
    models_names = []

    for model_name, model in models:
        start_time = time.time()
        accuracy = model.score(x_test, y_test)
        models_accuracies.append(accuracy)
        end_time = time.time()
        models_testing_time.append(abs(start_time - end_time))
        models_names.append(model_name)

    return models_accuracies, models_testing_time, models_names
