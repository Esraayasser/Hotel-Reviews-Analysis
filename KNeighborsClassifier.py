# region Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time
# endregion


def plot_different_k_values(x_train, y_train, x_test, y_test):
    error = []
    accuracy = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        prediction_i = knn.predict(x_test)
        error.append(np.mean(prediction_i != y_test))
        accuracy.append(np.mean(prediction_i == y_test) * 100)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 41), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 41), accuracy, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accuracy Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.show()
    return


def knn_train(x_train, y_train, k=3):
    start_time = time.time()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    end_time = time.time()
    return knn, abs(start_time-end_time)
