#!/usr/bin/env python

#################################################################
# In this programming assignment, I compared the performance 
# of Logistic Regression, Naive Bayes, Decision Tree, and Nearest 
# Neighbor on the Adult data set from the UCI repository 
# (http://archive.ics.uci.edu/ml/datasets/Adult). The prediction 
# task associated with this data set is to predict whether  
# or not a person makes more than $50K a year using census data.
##################################################################


import pandas as pd
import numpy as np
import operator
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
#import matplotlib.pyplot as plt



def pre_processing(df):	
	col_names = ['age','workclass','fnlwgt',\
	'education','education-num','marital-status','occupation','relationship','race',\
	'sex','capital-gain','capital-loss','hours-per-week','native-country','salary']
	df.columns = col_names
	
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


def split_x_y(df):
	index_x, index_y = df.columns[:-1], df.columns[-1]
	x = df[index_x]
	y = df[index_y]
	return x, y 
	

def logistic_regression(train,test):

	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)

	# use 10-fold validation to find optimal number of features   
	model = LogisticRegression()	
	rfecv = RFECV(estimator=model, step=1, cv=10,
              scoring='accuracy')   
	lgr = rfecv.fit(x_train,y_train)

	preds = lgr.predict(x_test)
	return metrics.accuracy_score(y_test, preds)
						

def naive_bayes(train,test):
	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)
	
	# prior feature selection might improve the performance
	
	gnb = GaussianNB()
	gnb = gnb.fit(x_train, y_train)
	
	preds = gnb.predict(x_test)
	return metrics.accuracy_score(y_test, preds)


def knn_classifier(train,test):
	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)
	
	best_model = [None,0]  # [model,score] 
			
	# here I perform 10-fold cross validation
	neighbors = list(range(1,20))
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
		if scores.mean() > best_model[1]:
			model = knn.fit(x_train,y_train)
			best_model = [model,scores.mean()]
		
	knn = best_model[0]
	preds = knn.predict(x_test)
	
	return metrics.accuracy_score(y_test, preds)


def decision_tree(train,test):
	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)
	
	best_model = [None,0]  # [model,score] 
	
	## here I perform 10-fold cross validation
	for i in [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]:   # test decision tree depth
		dct = tree.DecisionTreeClassifier(max_depth=i)
		scores = cross_val_score(dct, X=x_train, y=y_train, cv=10)
		if scores.mean() > best_model[1]:
			model = dct.fit(x_train,y_train)
			best_model = [model,scores.mean()]
	
	dct = best_model[0]
	preds = dct.predict(x_test)
	return metrics.accuracy_score(y_test, preds)


def main():
	train = pd.read_csv('adult.data',header=None)
	test = pd.read_csv('adult.test',header=None,skiprows=[0])
	outfile = open("output.txt","w")
	
	print "raw training set: %s, raw test set: %s" % (train.shape,test.shape)

	train_clean = pre_processing(train)
	test_clean = pre_processing(test)
	print "cleaned training set: %s, raw test set: %s" % (train_clean.shape,test_clean.shape)

	outfile.write("Statistics\n")

	nb_accuracy = naive_bayes(train_clean,test_clean)
	outfile.write("Naive Bayes accuracy: %f\n" % (nb_accuracy))

	knn_accuracy = knn_classifier(train_clean,test_clean)
	outfile.write("KNN accuracy: %f\n" % (knn_accuracy))
	
	dct_accuracy = decision_tree(train_clean,test_clean)
	outfile.write("Decision Tree accuracy: %f\n" % (dct_accuracy))
	
	lgr_accuracy = logistic_regression(train_clean,test_clean)
	outfile.write("Logistic Regression accuracy: %f\n" % (lgr_accuracy))
	
	outfile.close()



if __name__ == "__main__":
	main()

