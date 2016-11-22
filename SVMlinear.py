#Import Library

import csv
import random
import math
import operator
from sklearn import svm
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import KFold


def loadDataset(filename, trainingSet=[] , testSet=[], t1=[], t2=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for row in dataset:
			row[1],row[-1]=row[-1],row[1]
		for x in range(len(dataset)-1):
			for y in range(31) :
				dataset[x][y] = float(dataset[x][y])
			trainingSet.append(dataset[x])

		for row in trainingSet:
			if(len(row)!=0):
				t1.append(row[-1])
				del row[-1]

	return


def main():
	trainingSet=[]
	testSet=[]
	t1=[]
	t2=[]
	loadDataset('data.csv',trainingSet, testSet, t1, t2)
	print 'Train set: ' + repr(len(trainingSet))
	
	#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
	# Create SVM classification object 	
	# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next 		section.Train the model using the training sets and check score

	#predicted= model.predict(testSet)
	
	acc = 0
	folds = KFold(len(trainingSet),10)
	
	for itrain,itest in folds:
	    X_train = [trainingSet[i] for i in itrain]
	    Y_train = [t1[i] for i in itrain]
	    X_test = [trainingSet[i] for i in itest]
	    Y_test = [t1[i] for i in itest]
	    skm = svm.LinearSVC(C=1.0, penalty='l1', dual=False, max_iter=1000) 
	    skm.fit(X_train,Y_train)
	    acc = acc + skm.score(X_test,Y_test)
	    
	print 'Total Average Accuracy :',(acc/10.0)*100
	
	
main()
