'''
Ivan Christian

Homework 4

For this 

'''

'''
For testing purposes
'''
from utils.getimagenetclasses import test_parsesyn, testparse, test_parseclasslabel
import os

'''
Importing the classes from the custom and starting the task
'''
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
	if normalise == False:
		transformation = transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()])
	else:
		transformation = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	dataset = CustomLoader(im_directory, file, transformation = transformation)


	accuracy = train_custom(dataset)
	return accuracy

def transform_five_crops(im_directory, file):
	transformation = transforms.Compose([transforms.FiveCrop(224),lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(norm) for norm in norms])])
	dataset = CustomLoader(im_directory, file, crop_size=256, transformation = transformation)
	accuracy = train_custom(dataset)

	return accuracy

def train_custom(dataset):
	data_loader = DataLoader(dataset, batch_size=4, shuffle= True, num_workers=4)
	device = torch.device('cuda')
	'''
	For this case since the recommendation is not to use VGG or AlexNet, resnet will be used. DenseNet121 although less parameters is more memory hungry
	'''
	model = models.resnet18(pretrained=True).to(device)
	model.eval()

	correct = 0

	with torch.no_grad():
		for _, batch in enumerate(data_loader):
			images, labels = batch['image'], batch['label']
			images = images.to(device)
			labels = labels.to(device)

			output = model(images)
			pred = output.argmax(dim=1, keepdim=True)
			correct += pred.eq(labels.view_as(pred)).sum().item()

	accuracy = (correct / len(data_loader.dataset)) * 100
	return accuracy



def run():
	# confirmation = extract_filename('output.txt','val', 'data.csv')
	# print(confirmation)

	# '''
	# Need to do a bit more processing sincee the top row needs to be deleted
	# '''
	# with open("data.csv",'r') as f:
	# 	with open("finaldata.csv",'w') as f1:
	# 		next(f) # skip header line
	# 		for line in f:
	# 			f1.write(line)
	# os.remove("data.csv")
	# print('Finished Pre-processing the data. New data is saved in finaldata.csv')


	# '''
	# Problem 1: Normalised and not normalised
	# '''
	# accuracy_not_normalised = transform_normalise( 'imagenet2500', 'finaldata.csv', normalise = False)
	# accuracy_normalised = transform_normalise('imagenet2500','finaldata.csv', normalise = True)



	# print('\nProblem 1. Normalised/ Not Normalised Custom DataLoader: \n')
	# print(f'Accuracy for not normalised = {accuracy_not_normalised}')
	# print(f'Accuracy for normalised = {accuracy_normalised}')

	'''
	Problem 2 : 5 Crops
	'''
	accuracy_5_crop = transform_five_crops('imagenet2500','finaldata.csv')

	print('\nProblem 2. Five Crop Loader: \n')
	print(f'Accuracy for five crop = {accuracy_5_crop}')


if __name__=='__main__':
	run()