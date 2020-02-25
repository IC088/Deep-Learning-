'''
HW4 

Data Loader for the Imagenet dataset

'''


import os
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

'''
The custom dataset loader for this task

Custom Loader classs contains the necessary functions to finish the task


Init Input: 
- directory: string (directory name containing the actual imnage files)
- file : string  (filename containing the list of images)
- crop_size :  int (crop size of the image, defaulted to 224 to perform the center crop)
- transformation : (the transformation)
'''

class CustomLoader(Dataset):
	def __init__(self, im_directory, file, crop_size=224, transformation = None):
		self.to_tensor = transforms.ToTensor()
		self.im_directory = im_directory
		self.image_label = pd.read_csv(file, header= None)
		self.transformation = transformation
		self.crop_size = crop_size

	
	def _load(self, image_name):
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
		length = len(self.image_label)
		return length
	def __getitem__(self, index):
		'''
		get the image and label
		'''
		image_filename = os.path.join(self.im_directory, self.image_label.iloc[index,0])

		if type(self.image_label.iloc[index,0]) == float():
			print(self.image_label.iloc[index,0])

		image = self._load(image_filename)
		label = self.image_label.iloc[index,1]


		if self.transformation is not None:
			image = self._resize(image, self.crop_size)
			image = self.transformation(image)
		sample = {'image': image, 'label': label}
		return sample

