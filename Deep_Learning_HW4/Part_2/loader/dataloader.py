'''
Dataset class
'''

import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

import os
from PIL import Image
import numpy as np
import pandas as pd

class CustomFlowerDataset(Dataset):
	def __init__(self, image_dir, image_paths, labels, transformation = None):
		self.image_dir = image_dir
		self.labels = np.load(labels)
		self.image_label = self._load_paths(image_paths)
		self.transformation = transformation

	def _load_paths(self, path):
		'''
		Loading the image
		'''
		split_set = dict()

		with open(path) as f:
			lines = f.readlines()
			num_lines = len(lines)
			assert(num_lines == len(self.labels))
			for line_num in range(num_lines):
				full_image_path = os.path.join(self.image_dir, lines[line_num].strip('\n'))
				split_set[full_image_path] = self.labels[line_num]
		return pd.DataFrame.from_dict(split_set, orient='index')

	def _load_image(self, image_name):
		'''
		Load image after getting the image from the files. Used for this class operation
		Inputs: the image name 
		Outputs: Image from the array
		'''
		image = Image.open(image_name)
		image.load()
		image = np.array(image)


		if len(image.shape) == 2:
			image = np.expand_dims(image, 2)
			image = np.repeat(image, 3, 2)

		return Image.fromarray(image)

	def train_val_test_split(self, train_ratio, val_ratio):
		'''
		Splitting the train, val, test split
		'''
		dataset_len = len(self.image_label)
		train_len = int( train_ratio * dataset_len )
		val_len = int ( val_ratio * dataset_len )
		test_len = len(self) - (train_len + val_len)

		splits = [train_len, val_len, test_len]

		return random_split(self, splits) 

	def __len__(self):
		length = len(self.image_label)
		return length

	def __getitem__(self, index):
		'''
		similar to task 1
		'''
		image_path = self.image_label.index[index]
		image = self._load_image(image_path)
		if self.transformation is not None:
			image = self.transformation(image)
		label = self.labels[index]

		return {'image': image,'label': label}


