'''
Ivan Christian

Homework 4

'''

'''
For testing purposes
'''


from utils.getimagenetclasses import test_parsesyn, testparse, test_parseclasslabel


'''
Importing the classes from the custom and starting the task
'''


import os
import torch
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
'''
For the implementation of CustomLoader please see loader/dataloader.py
'''
from loader.dataloader import CustomLoader

from utils.name_pairs import extract_filename

'''
function load_dataset
'''

def transform_normalise(im_directory, file, normalise = False):
	'''
	For comparison purposes, it is checked whether it is normalised or not
	'''
	'''
	Function transform five_crops is a helper function to do five crops

	Input: 
	- im_directory : String (Directory of the image)
	- file : string ( image label file .csv format)
	- normalise : bool ( check whether the model needs to be normalised or not)

	Output:
	accuracy : float (accuracy of the model)
	'''

	if normalise == False:
		transformation = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()])
	else:
		transformation = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	dataset = CustomLoader(im_directory, file, transformation = transformation)


	accuracy = eval_custom(dataset)
	return accuracy

def transform_five_crops(im_directory, file,size=280):
	'''
	Function transform five_crops is a helper function to do five crops

	Input: 
	- im_directory : String (Directory of the image)
	- file : string ( image label file .csv format)
	- size : int (Specifying the size of the image)

	Output:
	accuracy : float (accuracy of the model)
	'''
	if size == 280:
		transformation = transforms.Compose([transforms.Resize(280), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])
	else:
		transformation = transforms.Compose([transforms.Resize(size), transforms.FiveCrop(size), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])


	dataset = CustomLoader(im_directory, file, crop_size=280, transformation = transformation)
	accuracy = eval_custom(dataset)

	return accuracy

def eval_custom(dataset):
	'''
	Function eval_custom is a helper function to evaluate the accuracy of the model

	input:
	dataset : dataset ( Dataset customised for this task) 

	output:
	accuracy : float (accuracy of the model)
	'''
	# Depending on how big your gpu memory is, can change the batch_size
	data_loader = DataLoader(dataset, batch_size=8, shuffle= True)
	torch.cuda.empty_cache()
	device = torch.device('cuda')
	'''
	For this case since the recommendation is not to use VGG or AlexNet, resnet will be used. DenseNet121 although less parameters is more memory hungry
	'''
	model = models.resnet18(pretrained=True).to(device)
	model.eval()

	correct = 0
	loops = 0
	with torch.no_grad():
		for _, batch in enumerate(data_loader):
			images, labels = batch['image'], batch['label']
			images = images.to(device)
			labels = labels.to(device)

			if images.size()[1] == 5:
				batch_size, num_crops, c, h, w = images.size()
				stacked = images.view(-1, c, h, w)

				output = model(stacked)
				result_avg = output.view(batch_size, num_crops, -1).mean(1)
				pred = result_avg.argmax(dim=1, keepdim=True)
				correct += pred.eq(labels.view_as(pred)).sum().item()
				
			else:
				output = model(images)
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(labels.view_as(pred)).sum().item()
			loops += 1
			
	accuracy = (correct / len(data_loader.dataset)) * 100

	return accuracy



def run():

	confirmation = extract_filename('output.txt','val', 'data.csv')
	print(confirmation)

	with open("data.csv",'r') as f:
		with open("finaldata.csv",'w') as f1:
			next(f) # skip header line
			for line in f:
				f1.write(line)
	os.remove("data.csv")
	print('Finished Pre-processing the data. New data is saved in finaldata.csv')


	'''
	Problem 1: Normalised and not normalised
	'''
	accuracy_not_normalised = transform_normalise( 'imagenet2500', 'finaldata.csv', normalise = False)
	accuracy_normalised = transform_normalise('imagenet2500','finaldata.csv', normalise = True)
	torch.cuda.empty_cache()


	print('\nProblem 1. Normalised/ Not Normalised Custom DataLoader: \n')
	print(f'Accuracy for not normalised = {accuracy_not_normalised}')
	print(f'Accuracy for normalised = {accuracy_normalised}')

	'''
	Problem 2 : 5 Crops
	'''
	accuracy_5_crop = transform_five_crops('imagenet2500','finaldata.csv')

	print('\nProblem 2. Five Crop Loader: \n')
	print(f'Accuracy for five crop = {accuracy_5_crop}')
	torch.cuda.empty_cache()
	
	'''
	Problem 3 : Bigger Sized Images
	'''

	accuracy_5_crop_bigger = transform_five_crops('imagenet2500','finaldata.csv', size=330)
	print('\nProblem 3. Different Input size for the NN: \n')
	print(f'Accuracy for five crop = {accuracy_5_crop_bigger}')
	torch.cuda.empty_cache()


if __name__=='__main__':
	run()