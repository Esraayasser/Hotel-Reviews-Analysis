from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time


def adaboost_model(x_train, y_train):
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME.R", n_estimators=100)

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train)

    start_time = time.time()

    model.fit(x_train, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    return model, training_time
