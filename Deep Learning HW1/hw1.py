'''
Deep Learning Homework 1 SUTD

Term 7
Ivan Christian
'''

import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset_directory = 'dataset'

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
	spring_shape = image_label_spring.shape[0]
	image_label_summer = image_label[image_label.summer != 0]
	summer_shape = image_label_summer.shape[0]
	image_label_autumn = image_label[image_label.autumn != 0]
	autumn_shape = image_label_autumn.shape[0]
	image_label_winter = image_label[image_label.winter != 0]
	winter_shape = image_label_winter.shape[0]
	image_label_data = pd.concat([image_label_spring, image_label_summer, image_label_autumn, image_label_winter]).drop_duplicates()
	x = np.array([np.load(os.path.join(os.path.join(dataset_directory,"imageclef2011_feats") , file + '_ft.npy')) for file in image_label_data.image])
	return x, image_label_data, spring_shape,summer_shape, autumn_shape, winter_shape
'''
function is used to split the train, test, val from the dataset
according to the requirement: 60 (train) - 15(test) - 25 (val)

Input: 	
x : training set in numpy array for later use
y : target set in pandas dataframe format for later use

'''
def split(x,y):
	x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.15, stratify=y)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=338/1150, stratify=y_train)
	return x_train, x_test, x_val, y_train, y_test, y_val

'''
Function is used to calculate the defined class wise accuracy. Since the propcessing for the dataset is 
Input:

y_pred : the predicted values of y from the model
y : labelled values of y from the dataset

Returns:

c_accuracy : 1/(total number of elements in the class) *sum of the correctly classified elements
'''


def class_wise_accuracy(y_pred, y, shape):
	c_accuracy = np.sum(y_pred * y) / ((shape/1353)*np.array(y).shape[0])
	return c_accuracy
'''
function write_results is used to write the results gotten from the homework in a txt file

Input:

results : dictionary containing the results

Returns:

None
'''
def write_results(results):

	import json

	if not os.path.exists(f'accuracy'):
		os.mkdir(f'accuracy')
	else:
		print('Folder exists, the next step will overwrite the results obtained')

	print('Writing out the results to folder accuracy. The results will be saved in "accFile.txt".')
	f = open(os.path.join('accuracy', "accFile.txt"), "w")   # 'r' for reading and 'w' for writing
	f.write(json.dumps(results))    # Write inside file 
	f.close() 

	print('Finished writing out file. Please check accfile.txt under the accuracy folder') 
	return

'''
function save is used to save the features in .npy files

Input:

x_train: x training data
x_test: x testing data
x_val: x validation data
y_train: y training data
y_test: y testing data
y_val: y validation data


Returns:

None
'''


def save(x_train, x_test, x_val, y_train, y_test, y_val):


	np.save(os.path.join('splits', "x_train.npy"), x_train)
	np.save(os.path.join('splits', "x_val.npy"), x_val)
	np.save(os.path.join('splits', "x_test.npy"), x_test)
	np.save(os.path.join('splits', "y_train.npy"), y_train)
	np.save(os.path.join('splits', "y_val.npy"), y_val)
	np.save(os.path.join('splits', "y_test.npy"), y_test)
	print('Features Saved!')



'''
Outline of the program:

Target Dataset (y) is preprocessed is by saving them in a pd dataframe which will contain the columns: 

['image'], ['spring'], ['summer'], ['autumn'], ['winter']


Once the the dataframe is created, each of the image ids (image files) will correspond to the features in the image feats folder and 
the relevant ones are saved to in another dataframe (x).

Column ['image'] is then dropped from y to create the labels for the seasons.

The dataset is then split into 60-15-25 in preparation for training.

For each column, which is denoted by the seasons, the features are trained with changing regularisation constant (c) to find the optimal one.
The optimal c is the c which returns the highest average class accuracy.

The optimal c is then used to predict test value after the model is retrained using the combination of training and validation set.

The results are then saved to the appropriate files.

'''


if __name__ == "__main__":


	x, target, spring_shape,summer_shape, autumn_shape, winter_shape = preprocess(filename[1], filename[0])

	seasons =  ['spring','summer', 'autumn', 'winter']

	seas_dict = {seasons[0]: spring_shape, seasons[1]: summer_shape, seasons[2]: autumn_shape, seasons[3]:winter_shape}

	regularization_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]

	'''
	splitting the dataset to find the optimal c
	'''
	y = target.drop(columns=['image']) #drop the image id since its no longer needed after the preprocessing to get the relevant features
	x_train, x_test, x_val, y_train, y_test, y_val = split(x,y)


	if not os.path.exists(f'splits'):
		os.mkdir(f'splits')
	else:
		print('Folder exists, replacing the old values with the new one.')
	save(x_train, x_test, x_val, y_train, y_test, y_val) # save the required data to .npy files. This can be found in the newly created splits folder
	'''
	Finding optimal c by finding the c which gives the majority best results
	'''
	best_c_list = [] 
	best_scores = []
	for season in seasons:
		scores = []

		for c in regularization_constants:
			svm = SVC(C=c, kernel = 'linear', probability=True)
			svm.fit(x_train,y_train[season])

			'''
			Analyse best c by calculating the class wise accuracy
			'''
			y_pred_class = svm.predict(x_val)

			acc = class_wise_accuracy(y_pred_class, y_val[season], seas_dict[season])
			scores.append(acc)

		best_c_list.append(regularization_constants[np.argmax(scores)])
		best_scores.append(np.max(scores))
		print(f'best c for {season} = {regularization_constants[np.argmax(scores)]} with score: {np.max(scores)}')


	best_c = max(set(best_c_list), key = best_c_list.count)
	average_scores = np.mean(best_scores)
	print(f'Best c : {best_c}, Average class wise accuracy tested against test set: {average_scores}')

	'''
	SVM: calculate the accuracy of the model
	'''

	x_train_val = np.concatenate((x_train, x_val)) # combine both training and validation set for x
	y_train_val = pd.concat((y_train, y_val)) # combine both training and validation set for y
	c_averaged = [] 
	c_concat = []

	for season in seasons:
		svm = SVC(C=best_c, kernel = 'linear', probability=True)
		svm.fit(x_train_val,y_train_val[season]) # Create model based on the new combined training set
		c_prelim_pred_prob = svm.predict_proba(x_test)
		c_concat.append(c_prelim_pred_prob[:,1])
		c_prelim_pred = np.zeros_like(c_prelim_pred_prob)
		c_prelim_pred[np.arange(len(c_prelim_pred_prob)), c_prelim_pred_prob.argmax(1)] = 1
		c_acc = class_wise_accuracy(c_prelim_pred[:,1],  y_test[season], seas_dict[season])
		c_averaged.append(c_acc)
		# print(f'{season} has class acc = {c_acc}')

	'''
	Calculating the average class wise accuracy
	'''
	c_averaged = np.mean(c_averaged)
	print(f'Overall average class wise accuracy = {c_averaged}')


	'''
	predict using vanilla accuracy
	'''

	y_vanilla_pred = np.array(c_concat).transpose()

	y_prelim_pred = np.zeros_like(y_vanilla_pred)

	y_prelim_pred[np.arange(len(y_vanilla_pred )), y_vanilla_pred.argmax(1)] = 1
	y_prelim_pred = y_prelim_pred.flatten()

	y_test_vanilla = np.array(y_test).flatten()

	vanilla_Acc = np.sum(y_prelim_pred == y_test_vanilla)/y_prelim_pred.shape[0]


	print(f'vanilla accuracy = {vanilla_Acc}')


	results = {'best c value': best_c, 'Average Class-wise accuracy': c_averaged, 'Vanilla Accuracy': vanilla_Acc}

	write_results(results)




