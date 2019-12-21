# importing necessary libraries
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import time


def svm_models(x_train, y_train):
    c_values = [1, 100000, 1000000]
    models = []
    times = []
    for c in c_values:
        start_time = time.time()
        model = svm.LinearSVC(C=c).fit(x_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time
        models.append(model, train_time)
    return models, times
