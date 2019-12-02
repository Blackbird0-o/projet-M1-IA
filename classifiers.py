# -------------Imports-------------
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


# -------------Classifiers-------------
def SVM(x_train, y_train, x_test, y_test):
  '''
  Support vecteur Machine classifieur
  '''

  clf_svc = SVC(gamma='auto', kernel='linear')
  model = clf_svc.fit(x_train, y_train)

  return model.predict(x_test)


def knn(x_train, y_train, x_test, y_test):
  '''
  KNN classifier
  '''

  clf = KNeighborsClassifier(n_neighbors=5)
  model = clf.fit(x_train, y_train)

  return model.predict(x_test)


def forest(x_train, y_train, x_test, y_test):
  '''
  Random forest classifier
  '''

  clf = RandomForestClassifier(
      n_estimators=1000, max_depth=4, random_state=0)
  model = clf.fit(x_train, y_train)

  return model.predict(x_test)


def maxiforest(x_train, y_train, x_test, y_test):
  '''
  Extra trees classifier
  '''

  clf = ExtraTreesClassifier(n_estimators=1000, max_depth=3, random_state=0)
  model = clf.fit(x_train, y_train)

  return model.predict(x_test)
