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

from sklearn.model_selection import train_test_split

dataset_directory = '.\\dataset'

path = os.listdir(dataset_directory)

'''
Get labels from the list of all labels
Saved as var labels in the form of an np.array
'''

labels = np.array([l.strip('\n').split('\t') for l in open(os.path.join(dataset_directory, 'concepts_2011.txt'))])
image_labels = np.array([i.strip('\n').split('.jpg ') for i in open(os.path.join(dataset_directory, 'trainset_gt_annotations.txt'))])

list_of_features = [os.path.join(os.path.join(dataset_directory,"imageclef2011_feats") , file) for file in os.listdir(os.path.join(dataset_directory,"imageclef2011_feats")) if file.endswith(".npy") ]
x = np.array([np.load(feat) for feat in list_of_features])

'''
Splitting the dataset to 60-15-25

'''

x_train, x_test, y_train, y_test = train_test_split()

x_train, x_val, y_train, y_val = train_test_split()
