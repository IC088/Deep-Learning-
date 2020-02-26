'''
Ivan Christian

Homework 4 - Task 2 Transfer Learrning
'''

import os

try:
	from tqdm import tqdm
except:
	print('Please run pip install tqdm since it\'s nice to have a progress bar')


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models


from loader.dataloader import CustomFlowerDataset
from utils.vis import visualise_train_val_loss
from utils.extract_filename import extract_name


def create_loaders(im_dir, im_paths, label, train_split=0.7, val_split=0.1):
	'''
	Function create_loaders is a function to create data loader for training

	Input:
	- im_dir : string
	- im_paths : string
	- label : 
	- train_split : float ()
	- val_split : float ()

	Output:
	- train_loader : DataLoader for training 
	- val_loader : DataLoader for validation
	- test_loader : DataLoader for testing
	'''

	transformation = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])

	dataset = CustomFlowerDataset(im_dir, im_paths, label, transformation = transformation)

	train_set, val_set, test_set = dataset.train_val_test_split(train_split, val_split)


	'''
	no need to make sure num workers = default since no lambda function exists here
	'''
	train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4)
	test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

	return train_loader, val_loader, test_loader


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

	model.train().to(device)

	train_losses = []

	correct = 0

	for index, batch in enumerate(train_loader):
		data = batch['image'].to(device)
		target = batch['label'].long().to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = loss_func(output, target)#CrossEntropyLoss
		loss.backward() #calculate the gradient 
		optimizer.step()


			# correct += (output == target).float().sum()
		train_losses.append(loss.item())


	train_loss = torch.mean((torch.tensor(train_losses)))

	print(f'Epoch: {epochs}')
	print(f'Training set: Average loss: {train_loss:.4f}')

	return train_loss

def validate ( model , device , val_loader , loss_func):
	'''
	Function validate is used to check accuracy against the validation set
	
	Input:
	- model
	- device
	- val_loader
	- loss_func

	Output:
	- val_loss
	- accuracy
	'''

	model.eval().to(device)
	val_loss = 0
	accuracy = 0

	print('Start Validation')

	with torch.no_grad():
		for _, batch in enumerate(val_loader):
			data = batch['image'].to(device)
			target = batch['label'].long().to(device)

			output = model(data)
			batch_loss = loss_func(output, target).item()#CrossEntropyLoss
			val_loss += batch_loss

			pred = output.argmax(dim=1, keepdim=True)
			accuracy += pred.eq(target.view_as(pred)).sum().item()
	val_loss /= len(val_loader)

	accuracy /= len(val_loader.dataset)

	print(f'Validation set: Average loss: {val_loss}, Accuracy: {accuracy}')
	return val_loss, accuracy

def test (model, device, test_loader):
	'''
	Function test is to test the model against the test set


	Inputs:
	- model
	'''

	print('Start Testing')
	model.eval()
	accuracy = 0
	with torch.no_grad():
		
		for _, batch in enumerate(test_loader):
			data = batch['image'].to(device)
			labels = batch['label'].long().to(device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim= True)
			accuracy += pred.eq(labels.view_as(pred)).sum().item()
	accuracy /= len(test_loader.dataset)
	return accuracy 


def custom_training(im_dir, im_paths, label, mode = 3, epochs = 10):

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(f'Starting Training with {device}.')

	if mode == 1:
		# mode 1 is to train from scracth
		model = models.resnet18(num_classes=102).to(device)
	elif mode == 2:
		# mode 2 is to do training the last few layers
		model = models.resnet18(pretrained=True).to(device)
	elif mode == 3:
		print('Loading model weights before training and training all layer.')
		# mode 3 is to do pre trained weights
		model = models.resnet18(pretrained=True).to(device)
		model.fc = nn.Linear(512, 102)

	else:
		print('Mode not recognised')
		return 


	print('Optimizer: SGD')

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)

	print('loss function : CrossEntropyLoss')
	loss_func = nn.CrossEntropyLoss().to(device)

	train_loader, val_loader, test_loader = create_loaders(im_dir, im_paths, label)

	print('Finished Loading Data to loader')

	train_losses_vis = []
	val_losses_vis = []

	for epoch in range(1, epochs + 1):
		train_loss = train(model, device, train_loader, optimizer, epoch, loss_func)
		val_loss, val_accuracy = validate(model, device, val_loader, loss_func)

		if (len(val_losses_vis) > 0) and (val_loss < min(val_losses_vis)):
			torch.save(model.state_dict(), 'hw4_model.pt')
		train_losses_vis.append(train_loss)
		val_losses_vis.append(val_loss)
		print(val_accuracy)

		print(f'Finished running epoch: {epoch}')
	
	print('Finished Training and Validation Process')

	test_accuracy = test(model, device, test_loader)

	print(f'Test Accuracy: {test_accuracy}')

	if mode == 1:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_full_training.png' )
	elif mode == 2:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_last2_layers.png' )
	elif mode == 3:
		visualise_train_val_loss(train_losses_vis, val_losses_vis, epochs, 'custom_pre_trained.png' )

def run():
	data = os.path.join('data', 'jpg')
	output = 'files.txt'
	label = os.path.join('labels', 'imagelabels.npy')
	confirmation = extract_name(data, 'jpg', output)

	print(confirmation)

	'''
	A
	'''

	custom_training(data, output, label, mode = 1, epochs = 80)
	torch.cuda.empty_cache()

	'''
	B
	'''
	custom_training(data, output, label, epochs = 80)
	torch.cuda.empty_cache()
	print('Finished Training ')

	


if __name__=='__main__':
	run()