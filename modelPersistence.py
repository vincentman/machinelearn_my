# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:37:15 2017

@author: vincentkao
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)
       
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l1', 
                          C=1000.0, 
                          random_state=0)

clf.fit(X_train, y_train) 
print('Accuray for training: ', clf.score(X_train, y_train))

# dump to persistence
from sklearn.externals import joblib
joblib.dump(clf, 'persist.pkl')

# load from persistence 
clf2 = joblib.load('persist.pkl') 
print(clf2.predict(X))
print('Accuray for test(load persistence): ', clf2.score(X, y))
