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
from sklearn.svm import LinearSVC

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
'''


def preprocess(dataset, reference, start=10, end=14):
	labels = pd.read_csv(os.path.join(dataset_directory, reference), sep="\t")
	image_labels = pd.read_csv(os.path.join(dataset_directory, dataset), sep=" ", header=None)
	image_label = image_labels.drop(columns = [i for i in range(1,start)])
	image_label = image_label.drop(columns = [i for i in range(end,len(image_labels.columns))])
	image_label.columns = ['image', 'spring','summer', 'autumn', 'winter']
	image_label_spring = image_label[image_label.spring != 0]
	image_label_summer = image_label[image_label.summer != 0]
	image_label_autumn = image_label[image_label.autumn != 0]
	image_label_winter = image_label[image_label.winter != 0]
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


x, target = preprocess(filename[1], filename[0])

seasons =  ['spring','summer', 'autumn', 'winter']
regularization_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]
best_c_list, average_score = [], []



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
	#print(f'x_train.shape: {x_train.shape}, x_test_shape: {x_test.shape}, x_val.shape: {x_val.shape}')
	#print(y_train.shape, y_test.shape, y_val.shape)

	if not os.path.exists(f'splits\\{season}'):
		os.mkdir(f'splits\\{season}')

	np.save(f'splits\\{season}\\x_train.npy', x_train)
	np.save(f'splits\\{season}\\x_val.npy', x_val)
	np.save(f'splits\\{season}\\x_test.npy', x_test)
	np.save(f'splits\\{season}\\y_train.npy', y_train)
	np.save(f'splits\\{season}\\y_val.npy', y_val)
	np.save(f'splits\\{season}\\y_test.npy', y_test)
	
	for i in regularization_constants:
		svm = LinearSVC(C=i)
		svm.fit(x_train,y_train)
		#print(f'constant {i}, season: {season}, score: {svm.score(x_val, y_val)}')
		scores.append(svm.score(x_val, y_val))
	best_c = regularization_constants[np.argmax(scores)]
	best_score = np.max(scores)
	average_score.append(best_score)
	best_c_list.append(best_c)


	# print("Best pair | c={}, score={}".format(best_c, best_score))
average_performance = np.mean(average_score)
print(f'The average best performance: {average_performance}')
'''
Conclusion: Best c is 0.01 for this project
'''
'''
SVM prediction
'''
total_accuracy = []

for season in seasons:
	y = target[season]
	x_train, x_test, x_val, y_train, y_test, y_val = split(x,y)
	x_train_val = np.concatenate((x_train, x_val))
	y_train_val = np.concatenate((y_train, y_val))
	svm = LinearSVC(C=np.mean(best_c_list))
	svm.fit(x_train_val, y_train_val)
	y_pred = svm.predict(x_test)
	'''
	Vanilla Accuracy
	'''

	accuracy = (1/len(y_pred)) * np.sum(np.floor((y_pred + np.array(y_test))/2) )
	print(f'The vanilla {season} accuracy: {accuracy}')
	total_accuracy.append(accuracy)
class_wise  = np.mean(total_accuracy)

print(f'class-wise averaged accuracy: {class_wise}')


