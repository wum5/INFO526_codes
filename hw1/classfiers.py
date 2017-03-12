#!/usr/bin/env python

#################################################################
# In this programming assignment, I compared the performance 
# of Logistic Regression, Naive Bayes, Decision Tree, and Nearest 
# Neighbor on the Adult data set from the UCI repository 
# (http://archive.ics.uci.edu/ml/datasets/Adult). The prediction 
# task associated with this data set is to predict whether  
# or not a person makes more than $50K a year using census data.

# Author: Meng Wu
# Date: March 1, 2017
##################################################################


import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


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
	# recursive feature elimination with cross-validation  	
	model = LogisticRegression()	
	rfecv = RFECV(estimator=model, step=1, cv=10, scoring='accuracy')   
	lgr = rfecv.fit(x_train,y_train)
	
	preds_prob = lgr.predict_proba(x_test)[:,1]
	fpr, tpr, thres = roc_curve(y_test, preds_prob)
	roc_auc = auc(fpr, tpr)

	preds = lgr.predict(x_test)
	accuracy = metrics.accuracy_score(y_test, preds)
	return accuracy, fpr, tpr, roc_auc						


def naive_bayes(train,test):
	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)
	
	# prior feature selection might improve the performance	
	gnb = GridSearchCV(GaussianNB(),{})
	gnb = gnb.fit(x_train, y_train)
	
	preds_prob = gnb.predict_proba(x_test)[:,1]
	fpr, tpr, thres = roc_curve(y_test, preds_prob)
	roc_auc = auc(fpr, tpr)
	
	preds = gnb.predict(x_test)
	accuracy = metrics.accuracy_score(y_test, preds)
	return accuracy, fpr, tpr, roc_auc	


def knn_classifier(train,test):
	x_train, y_train = split_x_y(train)
	x_test, y_test = split_x_y(test)
	
	# scale continuous features between 0 and 1
	x_train_minmax, x_test_minmax = scale_continuous(x_train,x_test)
                
	best_model = [None,0]  # [model,score] 
			
	# here I perform 10-fold cross validation
	neighbors = list(range(1,20))
	for k in neighbors:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn, x_train_minmax, y_train, cv=10, scoring='accuracy')
		if scores.mean() > best_model[1]:
			model = knn.fit(x_train_minmax,y_train)
			best_model = [model,scores.mean()]
		
	knn = best_model[0]
	
	preds_prob = knn.predict_proba(x_test_minmax)[:,1]
	fpr, tpr, thres = roc_curve(y_test, preds_prob)
	roc_auc = auc(fpr, tpr)
	
	preds = knn.predict(x_test_minmax)
	accuracy = metrics.accuracy_score(y_test, preds)
	
	return accuracy, fpr, tpr, roc_auc	
	

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
	
	preds_prob = dct.predict_proba(x_test)[:,1]
	fpr, tpr, thres = roc_curve(y_test, preds_prob)
	roc_auc = auc(fpr, tpr)
	
	preds = dct.predict(x_test)
	accuracy = metrics.accuracy_score(y_test, preds)
	
	return accuracy, fpr, tpr, roc_auc	


def histogram(df):
	x_train, y_train = split_x_y(df)
	continue_features = x_train.dtypes[(x_train.dtypes=="float64")|(x_train.dtypes=="int64")]
	
	x_train[continue_features.index.values].hist(figsize=[11,11])
	plt.savefig('Hist_continuous_features.pdf')
	plt.clf() 
	
	
def scale_continuous(x_train,x_test):
	min_max=MinMaxScaler()
	x_train_minmax=min_max.fit_transform(x_train[['age', 'capital-gain',
                'capital-loss', 'hours-per-week', 'fnlwgt','education-num']])
	x_test_minmax=min_max.fit_transform(x_test[['age', 'capital-gain',
                'capital-loss', 'hours-per-week', 'fnlwgt','education-num']])
	return x_train_minmax, x_test_minmax

	
	
def main():
	
	# import the data
	train = pd.read_csv('adult.data',header=None)
	test = pd.read_csv('adult.test',header=None,skiprows=[0])
	
	# remove entries with missing values
	train_clean = pre_processing(train)
	test_clean = pre_processing(test)
	print "raw training set: %s, raw test set: %s" % (train.shape,test.shape)
	print "cleaned training set: %s, raw test set: %s" % (train_clean.shape,test_clean.shape)

	# make histogram plot on the continuous features
	histogram(train_clean)
	
	# run the four classifiers
	gnb_accuracy, gnb_fpr, gnb_tpr, gnb_roc = naive_bayes(train_clean,test_clean)
	knn_accuracy, knn_fpr, knn_tpr, knn_roc  = knn_classifier(train_clean,test_clean)	
	dct_accuracy, dct_fpr, dct_tpr, dct_roc  = decision_tree(train_clean,test_clean)	
	lgr_accuracy, lgr_fpr, lgr_tpr, lgr_roc  = logistic_regression(train_clean,test_clean)

	# Write the accuracy to output file
	outfile = open("output.txt","w")
	outfile.write("Statistics\n")	
	outfile.write("Naive Bayes accuracy: %f\n" % (gnb_accuracy))
	outfile.write("KNN accuracy: %f\n" % (knn_accuracy))
	outfile.write("Decision Tree accuracy: %f\n" % (dct_accuracy))	
	outfile.write("Logistic Regression accuracy: %f\n" % (lgr_accuracy))
	outfile.close()		

	# Plot the ROC Curve
	plt.title('Receiver Operating Characteristic')		
	plt.plot(gnb_fpr, gnb_tpr, 'green', label='Naive Bayes (area=%0.2f)' % gnb_roc)
	plt.plot(lgr_fpr, lgr_tpr, color='blue', label='Logistic Regression (area=%0.2f)' % lgr_roc)
	plt.plot(dct_fpr, dct_tpr, 'red', label='Decision Tree (area=%0.2f)' % dct_roc)
	plt.plot(knn_fpr, knn_tpr, color='yellow', label='KNN (area=%0.2f)' % knn_roc)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],color='grey',linestyle='--')
	plt.xlim([0,1])
	plt.ylim([0,1])	
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')	
	plt.savefig('ROC_Curve.pdf')



if __name__ == "__main__":
	main()
