import csv
import random
import math
import operator
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score , recall_score, precision_score
def loadDataset(filename, split, trainingSet=[] , testSet=[], t1=[], t2=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for row in dataset:
			row[1],row[-1]=row[-1],row[1]
		for x in range(len(dataset)-1):
			for y in range(31) :
				dataset[x][y] = float(dataset[x][y])
			#if x<split*len(dataset):
			trainingSet.append(dataset[x])
			#else:
			#	testSet.append(dataset[x])
		
		for row in trainingSet:
			if(len(row)!=0):
				t1.append(row[-1])
				del row[-1]
			
		#for row in testSet:
		#	if(len(row)!=0):
		#		t2.append(row[-1])
		#		del row[-1]
	return

#def getAccuracy(t2, predictions,x):
#	correct = 0
#	for x in range(len(t2)):
#		if t2[x] == predictions[x]:
#			correct += 1
#	return (correct/float(x)) * 100.0  NO FUCKING NEED
	

def main():
	trainingSet=[]
	testSet=[]
	t1=[]
	t2=[]
	split = 0.67
	loadDataset('data.csv', split, trainingSet, testSet, t1, t2)
	print 'Train set: ' + repr(len(trainingSet))
	#print 'Test set: ' + repr(len(testSet))
	#predicted=[]
	#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
	# Create SVM classification object
 
	#model = linear_model.LogisticRegression(C=0.0000001, random_state=0, max_iter=100) 
		
	# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next 		section.Train the model using the training sets and check score
	
	#model.fit(trainingSet, t1)
	
	
	acc = 0
	folds = KFold(len(trainingSet),10)
	f1=0 
	recall=0
	precision=0

	for itrain,itest in folds:
	    X_train = [trainingSet[i] for i in itrain]
	    Y_train = [t1[i] for i in itrain]
	    X_test = [trainingSet[i] for i in itest]
	    Y_test = [t1[i] for i in itest]
	    skm = LR(C=1.0,penalty='l1')
	    skm.fit(X_train,Y_train)
 	    Ypred=skm.predict(X_test)				
	    acc = acc + skm.score(X_test,Y_test)
	    f1=f1+ f1_score(Ypred, Y_test, pos_label='B')
            recall=recall+ recall_score(Ypred, Y_test, pos_label='B')
	    precision=precision+precision_score(Ypred, Y_test, pos_label='B')		
	print 'Total Average Accuracy :',(acc/10.0)*100
	print 'f1 score:', (f1/10.0)
	print 'recall score:', (recall/10.0)
	print 'precision score:', (precision/10.0)
	
	#model.score(trainingSet, t1)
	#print "lols"

	#predicted= model.predict(testSet)
	
	#for x in range(len(predicted)):
	#	result = predicted[x]
	#	print('> predicted=' + repr(result) + ', actual=' + repr(t2[x]))
	#accuracy = getAccuracy(t2, predicted, len(testSet))
	#print('Accuracy: ' + repr(accuracy) + '%')
	
	
main()
