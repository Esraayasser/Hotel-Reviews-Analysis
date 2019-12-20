# importing necessary libraries
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import time


def svm_one_versus_rest_training(x_train, y_train):
    start_time = time.time()

    # training a linear SVM classifier
    svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(x_train, y_train)

    end_time = time.time()
    return svm_model_linear_ovr, abs(start_time-end_time)


def svm_one_versus_one_training(x_train, y_train):

    start_time = time.time()

    # training a linear SVM classifier
    svm_model_linear_ovo = SVC(kernel='linear', C=1).fit(x_train, y_train)

    end_time = time.time()
    return svm_model_linear_ovo, abs(start_time-end_time)
