# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:41:19 2017

@author: vincentkao
"""

from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# LogisticRegression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=1000.0, random_state=0)
log.fit(X_train_std, y_train)
y_pred = log.predict(X_test_std)
print("Accuracy for LogisticRegression: %.2f" % accuracy_score(y_test, y_pred))

# GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std, y_train)
y_pred = gnb.predict(X_test_std)
print("Accuracy for GaussianNB: %.2f" % accuracy_score(y_test, y_pred))

# MultinomialNB: training data can't be negative
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print("Accuracy for MultinomialNB: %.2f" % accuracy_score(y_test, y_pred))

# BernoulliNB
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train_std, y_train)
y_pred = bnb.predict(X_test_std)
print("Accuracy for BernoulliNB: %.2f" % accuracy_score(y_test, y_pred))