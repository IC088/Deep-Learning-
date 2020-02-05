'''
Deep Learning Homework 1 SUTD

Term 7
Ivan Christian
'''


'''
Question 1: 
What is the difference to a random 60−15−25 split of the whole data as compared to split 
class-wise? 

- 
Why I asked you to split class-wise ? 
Explain in at most 5 sentences.




'''

import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset_directory = '.\\dataset'

path = os.listdir(dataset_directory)

'''
Data Preprocessing:

Filter out the unnecessary data from the original dataset for training in the following parts

'''
filename = ['concepts_2011.txt', 'trainset_gt_annotations.txt']

'''
function preprocess is used to preprocess the dataset

This is done to filter the image label for the seasons

Input: 	dataset = dataset annotations in csv/txt format as provided by the project
		reference = reference input is used to indicate the code for the table category
		start = indicate which column should be left in dataset by default is 10 for spring
		end = indicate which column should be left in dataset by default is 14 for winter

Ouput:

Returns

x : training data from the dataset
image_label_data : target (true labels of the classification)
'''


def preprocess(dataset, reference, start=10, end=14):
	labels = pd.read_csv(os.path.join(dataset_directory, reference), sep="\t")
	image_labels = pd.read_csv(os.path.join(dataset_directory, dataset), sep=" ", header=None)
	image_label = image_labels.drop(columns = [i for i in range(1,start)])
	image_label = image_label.drop(columns = [i for i in range(end,len(image_labels.columns))])
	image_label.columns = ['image', 'spring','summer', 'autumn', 'winter']
	image_label_spring = image_label[image_label.spring != 0]
	# spring_shape = image_label_spring.shape
	image_label_summer = image_label[image_label.summer != 0]
	# summer_shape = image_label_summer.shape
	image_label_autumn = image_label[image_label.autumn != 0]
	# autumn_shape = image_label_autumn.shape
	image_label_winter = image_label[image_label.winter != 0]
	# winter_shape = image_label_winter.shape
	image_label_data = pd.concat([image_label_spring, image_label_summer, image_label_autumn, image_label_winter]).drop_duplicates()
	x = np.array([np.load(os.path.join(os.path.join(dataset_directory,"imageclef2011_feats") , file + '_ft.npy')) for file in image_label_data.image])
	return x, image_label_data
'''
function is used to split the train, test, val from the dataset
according to the requirement: 60 (train) - 15(test) - 25 (val)

Input: 	x = training set
		y = target set

'''
def split(x,y):
	x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.15, stratify=y)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=338/1150, stratify=y_train)
	return x_train, x_test, x_val, y_train, y_test, y_val

'''
function is used to calculate the defined class wise accuracy. Since the propcessing for the dataset is 
'''


def class_wise_accuracy(y_pred, y):
	# accuracy = np.sum(y_pred == y)
	return np.sum(y_pred == y) /np.array(y).shape




x, target = preprocess(filename[1], filename[0])

seasons =  ['spring','summer', 'autumn', 'winter']

#seasons = ['spring']
# class_total = {'spring': spring_shape,'summer': summer_shape, 'autumn': autumn_shape,'winter': winter_shape}
regularization_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]
#regularization_constants = [0.01]


regu_dict = { 0.01: [], 0.1: [], 0.1**0.5: [], 1: [], 10**0.5: [], 10: [], 100**0.5: []  }





'''
Deciding the best c for each train score
'''

for season in seasons:
	y = target[season]
	scores = []
	'''
	Splitting the dataset to 60-15-25 per class
	'''

	x_train, x_test, x_val, y_train, y_test, y_val = split(x,y)

	if not os.path.exists(f'splits\\{season}'):
		os.mkdir(f'splits\\{season}')

	np.save(f'splits\\{season}\\x_train.npy', x_train)
	np.save(f'splits\\{season}\\x_val.npy', x_val)
	np.save(f'splits\\{season}\\x_test.npy', x_test)
	np.save(f'splits\\{season}\\y_train.npy', y_train)
	np.save(f'splits\\{season}\\y_val.npy', y_val)
	np.save(f'splits\\{season}\\y_test.npy', y_test)
	
	for c in regularization_constants:
		svm = SVC(C=c, kernel = 'linear', probability=True)
		svm.fit(x_train,y_train)

		'''
		Analyse best c by calculating the class wise advantage
		'''
		y_pred = svm.predict(x_val)

		acc = class_wise_accuracy(y_pred, y_val)
		scores.append(acc)
		regu_dict[c].append(acc)

for i in regu_dict:
	print(f'constant {i} :  averaged accuracy: {np.mean(regu_dict[i])}')


'''
Conclusion best c is 0.01
'''

# y_pred = []

# seasons = ['spring']


# for season in seasons:
# 	y = target[season]
# 	# y = y.drop(columns = ['image'])

# 	x_train, x_test, x_val, y_train, y_test, y_val = split(x,y)
# 	x_train_val = np.concatenate((x_train, x_val))
# 	y_train_val = np.concatenate((y_train, y_val))

# 	svm = SVC(C=0.01, kernel = 'linear', probability=True)
# 	svm.fit(x_train_val, y_train_val)

# 	# Get probability that it is true for designated season


# 	prelim_pred = svm.predict_proba(x_test)[:,1] # returns probability for  0: index 0, 1: index 1

# 	print(prelim_pred)
	# prelim_pred = svm.predict_proba(x_test)[:,1]

	# print(y_test)
	# acc = class_wise_accuracy(prelim_pred, y_test)

	# print(f'{season} : {acc}')

	# y_pred.append(prelim_pred)

# y_pred = np.vstack(y_pred)


# print(y_pred.transpose())

# print(y_pred)
# print(f'{season} : {y_pred}')

# print(y_pred)

# print(y_pred)

# 	c_acc = class_wise_accuracy(y_pred, y_test)
# 	print(f'{season} accuracy: {c_acc}')
# 	final_acc.append(c_acc)

# average_acc = np.mean(final_acc)

# print(f'the average accuracy: {average_acc}')

# 	# print("Best pair | c={}, score={}".format(best_c, best_score))
# average_performance = np.mean(average_score)
# print(f'The average best performance: {average_performance}')
# '''
# Conclusion: Best c is 0.01 for this project
# '''
# '''
# SVM prediction
# '''
# total_accuracy = []

# for season in seasons:
# 	y = target[season]
# 	x_train, x_test, x_val, y_train, y_test, y_val = split(x,y)
# 	x_train_val = np.concatenate((x_train, x_val))
# 	y_train_val = np.concatenate((y_train, y_val))
# 	svm = LinearSVC(C=np.mean(best_c_list))
# 	svm.fit(x_train_val, y_train_val)
# 	y_pred = svm.predict(x_test)
# 	'''
# 	Vanilla Accuracy
# 	'''

# 	accuracy = (1/len(y_pred)) * np.sum(np.floor((y_pred + np.array(y_test))/2) )
# 	print(f'The vanilla {season} accuracy: {accuracy}')
# 	total_accuracy.append(accuracy)
# class_wise  = np.mean(total_accuracy)

# print(f'class-wise averaged accuracy: {class_wise}')

