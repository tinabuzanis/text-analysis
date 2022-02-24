

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Imports / Setup     --
#···············································································
import math
import random
import pandas as pd
from sklearn.datasets import load_iris

# Easier to just work with data directly instead of using df
inp = load_iris()
# temp = {}
# temp['data'] = data['data'].to[]
# temp['target'] = data['target'].to[]
# df = pd.DataFrame(temp)

data = inp['data']
target = inp['target']

# resulting dataset is [*values, class]
res = []
for i, _ in enumerate(data):
    res.append([*data[i], target[i]])


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────




## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --          --
#···············································································
from random import randrange
from math import sqrt
from math import exp
from math import pi
 
# implementation of cv_split
# takes the dataset and requested num of folds
def cv_split(dataset, num_folds=3):
	dataset_split = []
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / num_folds) # divides into folds
	for _ in range(num_folds):
		fold = []
		while len(fold) < fold_size: # for each fold 
			index = randrange(len(dataset_copy)) # take a random index of what's remaining
			fold.append(dataset_copy.pop(index)) # and add it to the current fold
		dataset_split.append(fold) # concat all of the folds
	return dataset_split
 
# calculate accuracy
def accuracy(actual, pred):
	correct = 0
	for i in range(len(actual)): # for all of the preds
		if actual[i] == pred[i]: # if the real value matches the predicted
			correct += 1 # note that we got it right
	return correct / float(len(actual)) * 100.0 # scale to be out of 100
 
# eval with cv_split
def eval_scores(dataset, num_folds) -> list:
	folds = cv_split(dataset, num_folds) # get the folds
	scores = [] 
	for fold in folds: # for each fold
		train_set = list(folds) # get the list of folds that we got from cv_split
		train_set.remove(fold) # remove the fold of this iter (we will add it to test_set)
		train_set = sum(train_set, []) # flatten
		test_set = [] 
		for row in fold: # for each row in the current fold
			row_copy = list(row) # copy the row as a list
			test_set.append(row_copy) # and append it to rest_set
			row_copy[-1] = None # remove the class_val column
		pred = naive_bayes(train_set, test_set) # run Naive Bayes
		actual = [row[-1] for row in fold] # get what the prediction should be
		acc = accuracy(actual, pred) # calculate accuracy
		scores.append(acc)
	return scores
 
# sort the dataset by the class values
def records_by_class(dataset) -> dict:
	records = {}
	for i in range(len(dataset)): # for all rows in dataset
		record  = dataset[i] # take the current row
		class_val = record[-1] # ignore the class val
		if (class_val not in records): # if we haven't seen this class_val yet
			records[class_val] = [] # we have a new key
		records[class_val].append(record) # add all of the records from that class_val to records
	return records
 
def mean(nums):
	return sum(nums)/float(len(nums))
 
def stdev(nums):
	avg = mean(nums)
	variance = sum([(x-avg)**2 for x in nums]) / float(len(nums)-1)
	return sqrt(variance)

# mean and std of records 
def dataset_stats(dataset):
    # iterate over each row and get the mean, std and len for each 
	stats = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(stats[-1]) # we don't care about the label right now
	return stats
 
# calculates stats per class
def class_stats(dataset):
	records = records_by_class(dataset)
	stats = {}
	for class_val, rows in records.items():
		stats[class_val] = dataset_stats(rows)
	return stats
 
# gaussian dist. probs
def calculate_probs(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# probs that a given row belongs to each class
def class_probs(stats, row):
    # num of training records
	total_rows = sum([stats[label][0][2] for label in stats])
	probs = {} # holds prob tht element belongs to class 
	for class_val, class_stats in stats.items(): # calcualte probs
		probs[class_val] = stats[class_val][0][2]/float(total_rows)
		for i in range(len(class_stats)):
			mean, stdev, _ = class_stats[i]
			probs[class_val] *= calculate_probs(row[i], mean, stdev)
	return probs
 
# make preds
def predict(stats, row):
	probs = class_probs(stats, row) # get the probs that a row belongs to a given class
	best_label, best_prob = None, -1 
	for class_val, prob in probs.items():  # get most likely label
		if best_label is None or prob > best_prob:
			best_prob = prob
			best_label = class_val
	return best_label
 
# naive bayes
def naive_bayes(train, test):
	stats = class_stats(train)
	preds = []
	for row in test:
		output = predict(stats, row) # predict using test stats
		preds.append(output)
	return(preds)
 
num_folds = 3
scores = eval_scores(res, num_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
dataset
