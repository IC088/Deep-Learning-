'''
Ivan Christian

DL Small Project
'''

from loader.loader import CustomMultiLabelLoader
from utils.vis import visualise_from_pickle, tailAccuracy,show_images, visualise_loss_accuracy
from utils.vocparseclslabels import PascalVOC


try:
	from tqdm import tqdm
	from pprint import pprint
except:
	print('Please install tqdm >>>> pip install tqdm')


import os
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from PIL import Image

from sklearn.metrics import average_precision_score


def create_loader(path,dev=False, transformation = None):
	'''
	Function to create loaders

	Input:
	- path : String ( path to teh images and labels )
	- dev : Boolean ( Mode of the training, defaulted to False. Done to test if the code is working properly or not)
	Outputs:
	- train_loader : DataLoader ( Training DataLoader )
	- val_loader : DataLoader ( Validation dataloader )
	- pos_weights :list ( postive weights of the classifications )
	'''

	# transformation = transforms.Compose([transforms.CenterCrop(280), transforms.ToTensor()])
	# transformation = transforms.Compose([transforms.Resize(size), transforms.FiveCrop(size), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])
	train_dataset = CustomMultiLabelLoader(path, train_or_val = 'train' , dev = dev, transformation = transformation)
	val_dataset = CustomMultiLabelLoader(path, train_or_val = 'val' , dev = dev, transformation = transformation)
	#same dataset but different purpose
	test_dataset = CustomMultiLabelLoader(path, train_or_val = 'val' , dev = dev, transformation = transformation)
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
	pos_weights = train_dataset.pos_weights
	return train_loader, val_loader, pos_weights, test_dataset


def train( model, device , train_loader , optimizer, epochs, loss_func ):
	'''
	Function train is a function to train the model
	input: 
	- model : model
	- device : device that is being used
	- train_loader : data loader for training
	- optimizer : which optimizer used
	- epoch : at what epoch this is runnning in
	- loss_func : loss function used
	Output:
	- train_loss : float ( average train loss )
	'''

	print('Starting Training')

	model.train().to(device)

	train_losses = []

	correct = 0

	for index, batch in enumerate(train_loader):
		data = batch['image'].to(device)
		target = batch['label'].float().to(device)
		optimizer.zero_grad()

		if len(data.size()) == 5:

			batch_size, num_crops, c, h, w = data.size()
			stacked = data.view(-1, c, h, w)
			output = model(stacked)
			output,_ = output.view(batch_size, num_crops, -1).max(1)
		else:
			output = model(data)


		loss = loss_func(output, target)
		loss.backward() #calculate the gradient 
		optimizer.step()

		train_losses.append(loss.item())


	train_loss = torch.mean((torch.tensor(train_losses)))

	print(f'Epoch: {epochs}\r')
	print(f'Training set: Average loss: {train_loss:.4f}\r')

	return train_loss

def validate ( model , device , val_loader , loss_func ):
	'''
	Function validate is used to check accuracy against the validation set
	
	Input:
	- model: model
	- device: string ( 'cuda' or 'cpu')
	- val_loader: DataLoader (validation data loader)
	- loss_func: (loss function chosen)

	Output:
	- val_loss: float (validation loss)
	- precision: float (validation accuracy)
	'''

	model.eval().to(device)
	val_loss = 0

	ls_pred = []
	ls_target = []

	print('Start Validation')

	with torch.no_grad():
		for _, batch in enumerate(val_loader):
			data = batch['image'].to(device)
			target = batch['label'].float().to(device)


			if len(data.size()) == 5:

				batch_size, num_crops, c, h, w = data.size()
				stacked = data.view(-1, c, h, w)
				output = model(stacked)
				output,_ = output.view(batch_size, num_crops, -1).max(1)
			else:
				output = model(data)


			batch_loss = loss_func(output, target).item()
			val_loss += batch_loss

			pred = torch.sigmoid(output) # binary for tail accuracy

			ls_pred += pred.tolist() # To find the average score precision
			ls_target += target.int().tolist()

	ls_pred = np.array(ls_pred)
	ls_target = np.array(ls_target)
	ap = [average_precision_score(ls_target[:,i],ls_pred[:,i]) for i in range(20)]
	precision = np.mean(ap)
	val_loss /= len(val_loader)

	print(f'Validation set: Average loss: {val_loss}, Precision: {precision}')
	return val_loss, precision



def load_image ( path, name , transformation, device ='cuda'):
	'''
	Function is used to load test images ( taken from the validation set ) to tensor for input. Helper function

	Input : 
	- path : string (root path for dataset)
	- name : string (image names)
	- transformation : Transform ( Transformation for the data --> Augmentation)
	- device :
	Output :
	- images : float tensor in cuda/device
	'''

	images = Image.open(os.path.join(path, 'JPEGImages' ,name))
	images = transformation(images).float()

	return images.to(device)

def image_scores(path, models, model_path, output_path, test_dataset, transformation, device = 'cuda'):
	'''
	Function is used to calculate the image prediction scores

	Input :
	- path : string (root directory for the dataset)
	- models : model (model used in this task)
	- model_path : string ( model directory )
	- test_dataset : Dataset ( essentially the testing files )
	- transformation : Transform ( Augmentation )
	- device : string ('cuda')
	'''

	print(f'Loading model to score images. Scores saved {output_path}')


	model_file = torch.load(model_path)
	models.load_state_dict(model_file)

	models.to(device)
	models.eval()
	score_list = []

	with tqdm(total = len(test_dataset.images)) as bar:
		for name , labels in zip(test_dataset.images, test_dataset.labels):
			images = load_image(path, name, transformation)
			outputs,_ = models(images).max(0)

			outputs = torch.sigmoid(outputs)

			score_list.append((name, outputs))
			bar.update()

	pprint(score_list[:5])

	import pickle

	with open(output_path, 'wb') as f:
		pickle.dump(score_list,f)


def custom_training(path,train_loader, val_loader,test_dataset,pos_weights, transformation, epochs=5):
	'''
	Function for custom task training. 

	Input:
	- train_loader : DataLoader ( training dataset Loader )
	- val_loader : DataLoader( Validation Data Loader)
	- pos_weights : list ( list of the positive weights to make sure that the 1's get more recognition)
	- epochs : int ( number of epochs for training. Defaulted to 5 )
	Output:
	- filename : String ( Finished training and validating is printed. Model is saved under 'small_project_model.pt')
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = models.resnet50(pretrained = True).to(device)

	def freeze_learning(layer):
		for param in layer.parameters():
			param.requires_grad = False

	freeze_learning(model.conv1) #Freeze the conv1 layer
	freeze_learning(model.bn1) #Freeze the bn1 layer
	freeze_learning(model.layer1) #Freeze the layer 1
	freeze_learning(model.layer2) #Freeze the layer 2
	freeze_learning(model.layer3) #Freeze the layer 3
	freeze_learning(model.layer4) #Freeze the layer 4
	model.fc = nn.Linear(2048, 20)

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	loss_func = nn.BCEWithLogitsLoss( pos_weight = torch.cuda.FloatTensor(pos_weights) ).to(device)

	train_losses_vis = []
	val_losses_vis = []
	val_accuracy_list = []
	filename = 'small_project_model.pt'
	scores_file = 'scores.txt'


	if os.path.isfile(filename):
		print ("File exists. Using Existing model")
	else:
		for epoch in range(1, epochs + 1):
			train_loss = train(model, device, train_loader, optimizer, epoch, loss_func)
			val_loss, val_accuracy = validate(model, device, val_loader, loss_func)
			if (len(val_losses_vis) > 0) and (val_loss < min(val_losses_vis)):
				torch.save(model.state_dict(), filename)
			train_losses_vis.append(train_loss)
			val_losses_vis.append(val_loss)
			val_accuracy_list.append(val_accuracy)

			print(f'Finished running epoch: {epoch}')

		print('Showing/Saving Tranining loss Graph')


		visualise_loss_accuracy(train_losses_vis, 'train_loss.png')
		
		print('Showing/Saving Validation loss Graph')

		visualise_loss_accuracy(val_losses_vis, 'val_loss.png')

		print('Showing/Saving Validation Accuracy Graph')

		visualise_loss_accuracy(val_accuracy_list, 'val_accuracy.png')
		
		print('Finished Training and Validation Process')

	print('Calculating image score')

	if os.path.isfile(scores_file):
		print ("File exists. Using existing Score file")
	else:
		image_scores(path,model, filename, scores_file, test_dataset, transformation)

	pascal = PascalVOC(path)

	print('Saving images')
	print('Generating top 50')
	top50images, top50scores = visualise_from_pickle(scores_file, path, test_dataset, pascal, mode = 1) # mode 1 : Top 50 images
	print('Generating bot 50')
	bot50images, bot50scores = visualise_from_pickle(scores_file, path, test_dataset, pascal, mode = 0) # mode 0 : Bottom 50 Images

	print('Generating Tail Accurac Graph')
	tailAccuracy(scores_file, path, test_dataset, pascal)

	print('Finished. Emptying CUDA cache')




def run():
	path = os.path.join('dataset','VOCdevkit','VOC2012')
	dev = False

	transformation = transforms.Compose([transforms.Resize(330), transforms.FiveCrop(224), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

	trainloader, valloader, pos_weights, test_dataset = create_loader(path, dev, transformation = transformation)

	custom_training(path, trainloader,valloader, test_dataset,pos_weights, transformation)
	torch.cuda.empty_cache()


if __name__=='__main__':
	run()