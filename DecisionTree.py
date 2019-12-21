from sklearn import tree
import time
from sklearn.tree import DecisionTreeClassifier


def decision_tree_model(X, Y):
    decisionTree = tree.DecisionTreeClassifier(max_depth=100)
    startTime = time.time()
    decisionTree.fit(X, Y)
    endTime = time.time()
    trainTime = abs(endTime - startTime)
    return decisionTree, trainTime