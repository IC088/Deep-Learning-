'''
Custom DataLoader for DL small Project
'''


import cv2
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split

import os

import pandas as pd
from PIL import Image
import numpy as np



class CustomMultiLabelLoader(Dataset):
	def __init__(self, root_dir,train_or_val = 'train', dev = False,transformation = None):
		self.transformation = transformation

		self.crop_size = 224

		'''
		taken from the code given
		'''
		self.root_dir = root_dir
		self.img_dir =  os.path.join(root_dir, 'JPEGImages')
		self.ann_dir = os.path.join(root_dir, 'Annotations')
		self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
		self.train_or_val = train_or_val
		self.classes = self.list_image_sets()
		self.images, self.image_names = self._list_images(self.set_dir, self.train_or_val)
		self.labels, self.pos_weights =self._list_labels(self.list_image_sets(), self.train_or_val)
		self.dev = dev


	def _list_images(self, trainval_path, train_or_val):
		new_trainval_file = os.path.join(trainval_path, f'{train_or_val}.txt')
		with open(new_trainval_file) as f:
			ls_images = [line.strip() + '.jpg' for line in f]
			ls_image_names = [line.strip() for line in f]
		return ls_images, ls_image_names
	def _list_labels( self, ls, train_or_val):
		labels = None
		lbls = []
		pos_w = [] 
		for c in ls:
			with open(os.path.join(self.set_dir, f'{c}_{train_or_val}.txt')) as f:
				temp = [int(line.strip().split()[1]) for line in f]
				pos_w.append((temp.count(-1)/temp.count(1)))
				lbls.append(temp)
		labels = (np.vstack(lbls).T + 1)/ 2


		return labels, pos_w



	def list_image_sets(self):
		"""
		Summary: 
			List all the image sets from Pascal VOC. Don't bother computing
			this on the fly, just remember it. It's faster.
		"""
		return [
			'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair',
			'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant',
			'sheep', 'sofa', 'train',
			'tvmonitor']

	def _imgs_from_category(self, dataset):
		"""
		Summary: 
		Args:
			cat_name (string): Category name as a string (from list_image_sets())
			dataset (string): "train", "val", "train_val", or "test" (if available)
		Returns:
			pandas dataframe: pandas DataFrame of all filenames from that category
		"""
		filename = os.path.join(self.set_dir, dataset + ".txt")
		df = pd.read_csv(
			filename,
			delim_whitespace=True,
			header=None,
			names=['filename', 'true'])
		return df

	def _imgs_as_list(self, dataset):
		"""
		Summary: 
			Get a list of filenames for images in a particular category
			as a list rather than a pandas dataframe.
		Args:
			cat_name (string): Category name as a string (from list_image_sets())
			dataset (string): "train", "val", "train_val", or "test" (if available)
		Returns:
			list of srings: all filenames from that category
		"""
		df = self._imgs_from_category( dataset)
		df = df[df['true'] == 1]
		return df['filename'].values
		'''
		End of copying
		'''
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

	def _resize(self, image, imsize):
		'''
		Resizing the image while keeping the ratio between the height and width. Used for this class
		This is simple crop method
		Inputs:
		- Image (the image file after being loaded )
		'''
		width, height = image.size
		scale = imsize / min(width,height)
		new_width = int(np.ceil(scale * width))
		new_height = int(np.ceil(scale * height))
		new_image = image.resize( (new_width, new_height))
		return new_image


	def __len__(self):
		length = len(self.labels)
		if self.dev:
			length = 100
			return length
		return length
	def __getitem__(self, index):
		image_path = os.path.join(self.img_dir, self.images[index])
		image = self._load_image(image_path)
		if self.transformation is not None:
			image = self._resize(image, self.crop_size)
			image = self.transformation(image)
		label = self.labels[index]
		return {'image': image,'label': label}