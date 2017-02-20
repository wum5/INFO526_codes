#!/usr/bin/env python

import pandas as pd
import numpy as np
import operator
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt


def pre_processing(df):	
	df = (df.replace(' ?',np.nan)).dropna()
	df['workclass']=df['workclass'].astype('category')
	df['education']=df['education'].astype('category')
	df['marital-status']=df['marital-status'].astype('category')
	df['occupation']=df['occupation'].astype('category')
	df['relationship']=df['relationship'].astype('category')
	df['race']=df['race'].astype('category')
	df['sex']=df['sex'].astype('category')
	df['native-country']=df['native-country'].astype('category')
	df['salary']=df['salary'].astype('category')
		
	cat_columns = df.select_dtypes(['category']).columns
	df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
	return df


def knn_classifier(train,test,k=1,cross_val=False):
	x_train, y_train = np.array(train.ix[:, :-1]), np.array(train['salary'])
	x_test, y_test = np.array(test.ix[:, :-1]), np.array(test['salary']) 
		
	# here is 10-fold validation
	if cross_val == True:
		myList = list(range(1,30))
		neighbors = filter(lambda x: x%2!=0, myList)
		knn_cv_MSE= {}
		for k in neighbors:
			knn = KNeighborsClassifier(n_neighbors=k)
			scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
			knn_cv_MSE[k] = 1-scores.mean()
		print knn_cv_MSE
		
	knn = KNeighborsClassifier(n_neighbors=20)
	knn.fit(x_train, y_train)
	preds = knn.predict(x_test)
	
	return metrics.accuracy_score(y_test, preds)


def decision_tree(train,test,depth=1,cross_val=False):
	x_train, y_train = np.array(train.ix[:, :-1]), np.array(train['salary'])
	x_test, y_test = np.array(test.ix[:, :-1]), np.array(test['salary']) 
	
	## Perform 7-fold corss validation 
	if cross_val == True:
		dt_cv_MSE = {}
		for i in range(1,20):
			clf = tree.DecisionTreeClassifier(max_depth=i)
			scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
			dt_cv_MSE[i] = 1-scores.mean()
		print dt_cv_MSE

	clf = tree.DecisionTreeClassifier(max_depth=depth)
	clf = clf.fit(x_train, y_train)
	
	preds = clf.predict(x_test)
	return metrics.accuracy_score(y_test, preds)


def main():
	train = pd.read_csv('adult.data',header=None,names=['age','workclass','fnlwgt',\
	'education','education-num','marital-status','occupation','relationship','race',\
	'sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])
	test = pd.read_csv('adult.test',header=None,skiprows=[0],names=['age','workclass',\
	'fnlwgt','education','education-num','marital-status','occupation','relationship',\
	'race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])
	print "raw training set: %s, raw test set: %s" % (train.shape,test.shape)

	train_clean = pre_processing(train)
	test_clean = pre_processing(test)
	print "cleaned training set: %s, raw test set: %s" % (train_clean.shape,test_clean.shape)

	knn_accuracy = knn_classifier(train_clean,test_clean,10)
	dt_accuracy = decision_tree(train_clean,test_clean,5)
	print "KNN(k=20) accuracy: %f" % (knn_accuracy)
	print "Decision tree(d=5) accuracy: %f" % (dt_accuracy)


if __name__ == "__main__":
	main()

