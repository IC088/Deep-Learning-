'''
Deep Learning Homework 1 SUTD

Term 7
Ivan Christian
'''


'''
Question 1: 
What is the difference to a random 60−15−25 split of the whole data as compared to split 
class-wise? 
Why I asked you to split classwise ? 
Explain in at most 5 sentences.


Answer:


'''

import os
import glob
import numpy as np
import pandas as pd

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
funciton preprocess for 

Input: dataset annotations in csv/txt format as provided by the 
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
	return image_label_data


y = preprocess(filename[1], filename[0])

print(y)
'''
Getting training
'''



'''
Splitting the dataset to 60-15-25 per class
'''



'''
SVM
'''
