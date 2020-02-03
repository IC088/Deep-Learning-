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
Reading the data using pandas dataframe

'''

labels = pd.read_csv(os.path.join(dataset_directory, 'concepts_2011.txt'), sep="\t")
image_labels = pd.read_csv(os.path.join(dataset_directory, 'trainset_gt_annotations.txt'), sep=" ", header=None)

image_labels[0] = image_labels[0].map(lambda x: x.strip('.jpg'))

list_of_features = [os.path.join(os.path.join(dataset_directory,"imageclef2011_feats") , file) for file in os.listdir(os.path.join(dataset_directory,"imageclef2011_feats")) if file.endswith(".npy") ]

'''
Features for x
'''

x = np.array([np.load(feat) for feat in list_of_features])


'''
Splitting the dataset to 60-15-25 per class
'''



'''
SVM
'''


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

regularization_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]
scores = []

for c in regularization_constants:
    svm = OneVsRestClassifier(estimator=SVC(C=c, kernel="linear"))
    svm.fit(x_train, y_train)
    scores.append(svm.score(x_val, y_val))
    
best_c = regularization_constants[np.argmax(scores)]
best_score = np.max(scores)
print("Best pair | c={}, score={}".format(best_c, best_score))