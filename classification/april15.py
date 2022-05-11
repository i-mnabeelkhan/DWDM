# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


'''making classification dataset randomly'''
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)

'''splitting in test and train'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

'''generating logistic regression classifier'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)

'''generating decision tree classifier'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
#by default gini index is used, if you want to use entropy
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


'''exporting and prinintg decision tree'''
from sklearn.tree.export import export_text
tree.plot_tree(clf,max_depth=1)
r = export_text(clf)
print(r)

'''generating random forest classifier'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()


'''generating naive bayes classifier'''
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

'''generating support vector machine classifier'''
from sklearn.svm import SVC
clf = SVC(gamma='auto')

'''scaling data, converting all the columns to smaller range of values like normalization but can only be applied if all columns are numerical'''
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.fit_transform(X_test)


'''fitting classifier -- training classifier'''
clf = clf.fit(X_train,y_train)
#predicting
y_pred=clf.predict(X_test)

'''making confusion matrix'''
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#how ROC curve is calculated, explore yourself 