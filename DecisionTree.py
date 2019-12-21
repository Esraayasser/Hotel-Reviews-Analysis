from sklearn import tree
import time


def decision_tree_model(X, Y):
    decision_tree = tree.DecisionTreeClassifier(max_depth=100)
    start_time = time.time()
    decision_tree.fit(X, Y)
    end_time = time.time()
    train_time = abs(end_time - start_time)
    return decision_tree, train_time
