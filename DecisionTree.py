import csv
import random
import math
import operator

class decisionnode:
   def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
	self.value=value
	self.results=results
	self.tb=tb
	self.fb=fb

# Divides a set on a specific column. Can handle numeric
# or nominal values
def divideset(rows,column,value):
	# Make a function that tells us if a row is in
	# the first group (true) or the second group (false)
	split_function=None
	if isinstance(value,int) or isinstance(value,float):
	 split_function=lambda row:row[column]>=value
	else:
	 split_function=lambda row:row[column]==value
	# Divide the rows into two sets and return them
	set1=[row for row in rows if split_function(row)]
	set2=[row for row in rows if not split_function(row)]
	return (set1,set2)

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for row in dataset:
		row[1],row[-1]=row[-1],row[1]
	    for x in range(len(dataset)-1):
	        for y in range(31) :
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])



# Create counts of possible results (the last column of
# each row is the result)
def uniquecounts(rows):
    results={}
    for row in rows:
	# The result is the last column
	r=row[len(row)-1]
	if r not in results: results[r]=0
	results[r]+=1
    return results

# Entropy is the sum of p(x)log(p(x)) across all
# the different possible results
def entropy(rows):
	from math import log
	log2=lambda x:log(x)/log(2)
	results=uniquecounts(rows)
	# Now calculate the entropy
	ent=0.0
	for r in results.keys( ):
		p=float(results[r])/len(rows)
		ent=ent-p*log2(p)
	return ent

def buildtree(rows,scoref=entropy):
	if len(rows)==0: return decisionnode( )
	current_score=scoref(rows)

	# Set up some variables to track the best criteria
	best_gain=0.0
	best_criteria=None
	best_sets=None

	column_count=len(rows[0])-1
	for col in range(0,column_count):
	  # Generate the list of different values in
	  # this column
	  column_values={}
	  
	  for row in rows:
		column_values[row[col]]=1
	  
	  # Now try dividing the rows up for each value
	  # in this column
       	  for value in column_values.keys( ):
		(set1,set2)=divideset(rows,col,value)
		
		# Information gain
		p=float(len(set1))/len(rows)
		gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
		if gain>best_gain and len(set1)>0 and len(set2)>0:
		  best_gain=gain
		  best_criteria=(col,value)
		  best_sets=(set1,set2)
	   # Create the subbranches
	if best_gain>0:
		trueBranch=buildtree(best_sets[0])
		falseBranch=buildtree(best_sets[1])
		return decisionnode(col=best_criteria[0],value=best_criteria[1],
		tb=trueBranch,fb=falseBranch)
        else:
		return decisionnode(results=uniquecounts(rows))

def classify(observation,tree):
 if tree.results!=None:
	return tree.results.keys()
 else:
	v=observation[tree.col]
	branch=None
	if isinstance(v,int) or isinstance(v,float):
	 if v>=tree.value: branch=tree.tb
	 else: branch=tree.fb
        else:
	 if v==tree.value: branch=tree.tb
	 else: branch=tree.fb
 return classify(observation,branch)

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('data.csv', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))
	predictions=[]
	
	tree=buildtree(trainingSet);	

	for x in range(len(testSet)):
		result = classify(testSet[x],tree);
		predictions.append(result[0])
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()

 
 
	



