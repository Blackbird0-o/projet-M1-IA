# -------------Imports-------------
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier


# -------------Classifiers-------------
def SVM(x_train, y_train, x_test, y_test):

    clf_svc = SVC(gamma='auto', kernel='poly')

    model = clf_svc.fit(x_train, y_train)

    return model.predict(x_test)


def forest(x_train, y_train, x_test, y_test):

    clf = RandomForestClassifier(
        n_estimators=1000, max_depth=4, random_state=0)

    model = clf.fit(x_train, y_train)

    return model.predict(x_test)


def maxiforest(x_train, y_train, x_test, y_test):

    clf = ExtraTreesClassifier(n_estimators=1000, max_depth=3, random_state=0)

    model = clf.fit(x_train, y_train)

    return model.predict(x_test)
