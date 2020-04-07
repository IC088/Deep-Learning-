'''
Deep Learning HW6


Ivan Christian 1003056
'''

import os
import string


from loader.custom_lstm import StarTrekLSTM
from loader.prep import get_data, STDataset
from utils.vis import write_to_text, plot_graph

import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import numpy as np



def create_loader(cat_ls, split, batch_size, all_letters, n_letters):
	'''
	Function to create Data loader baseed onthe Dataset class


	Inputs:
	- cat_ls : list (contains data)
	- split : float (ratio of test and train split)
	- batch_size : int (batch size)
	- all_letters : string (all allowed letteres, number, symbol)
	- n_letters : int (lenght of all_letters)


	Outputs:
	- train_loader : DataLoader (data loader for the training set)
	- test_loader : Dataloader (data for test set)
	- 


	'''
	def collate_fn(batch):

		data = batch[0][0][:-1] # no need EOS
		target = torch.Tensor(batch[0][1][1:]).long()
		return data, target

	train_list = cat_ls[split:]
	test_list = cat_ls[:split]

	train_dataset = STDataset(train_list, all_letters, n_letters)
	test_dataset = STDataset(test_list, all_letters, n_letters)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

	input_size = train_dataset.getInputSize()

	return train_loader, test_loader, input_size


def train(model, loss_func, device, train_loader, optimizer, epoch):
	'''
	Function to start Training

	Input:
	- model : LSTM (custom LSTM )
	- loss_func : loss function (Cross Entropy Loss)
	- device : cuda (defaulted to cuda for speed)
	- train_loader : DataLoader ( Train Data Loader )
	- epoch : int (current epoch number)

	Output:
	- train_loss : float (Average train loss)
	'''
	model.train()

	print(f'Start Training at epoch {epoch}')

	train_loss = 0
	samples = 0

	for data, char in train_loader:
		data = data.to(device).unsqueeze(0)
		char = char.to(device)

		optimizer.zero_grad()

		output = model(data)

		loss = loss_func(output, char)
		train_loss += loss.item()

		pred = output.argmax(dim=1)


		loss.backward()

		optimizer.step()
		samples += data.size()[0]


	train_loss /= samples

	print(f'Epoch {epoch}, Training Loss : {train_loss}')
	return train_loss


def test(model, loss_func, device, test_loader,epoch):
	'''
	Function to start test process

	Input:
	- model : lstm (Custom LSTM Model)
	- loss_func : loss function (Cross Entropy Loss)
	- device : cuda (defaulted to cuda for speed)
	- test_loader : DataLoader ( Test Data Loader )
	- epoch : int (current epoch number)


	Output:
	- test_accuracy : float (Averaged test accuracy)
	- test_loss : float (Average test loss)
	'''
	model.eval()

	print(f'Starting Test of epoch {epoch}')


	test_accuracy = 0
	samples = 0
	test_loss = 0


	with torch.no_grad():
		for data, char in test_loader:
			data = data.to(device).unsqueeze(0)
			char = char.to(device)


			output = model(data)
			loss = loss_func(output, char)

			test_loss += loss

			pred = output.argmax(dim=1)

			test_accuracy += torch.sum(pred==char).item()/(data.size()[1]-1)

			samples += data.size()[0]

	test_accuracy /= (samples)

	test_loss /= (samples)


	print(f'Epoch {epoch}, Average Test loss : {test_loss}, Test Accuracy : {test_accuracy}')


	return test_loss, test_accuracy


def custom_training(filename, all_letters, n_letters):

	'''
	Function to start the traininig process

	Input:
	- filename : string (text file name)
	- all_letters : string (all allowed letters, numbers, symbols)
	- n_letters : int (number of allowed letters, numbers, symbols)

	Output:
	- train_losses_vis: list (list of train losses for visualisation)
	- test_losses_vis: list
	- test_acc_vis: list 
	'''

	n_layers = 3 # As specified
	hidden_size = 200 # As specified
	dropout = 0.1 # As specified
	temp = 0.5 #As specified

	lr = 0.01 
	batch_size = 32

	test_split = 0.2

	epochs = 40

	all_samples = 0
	out_filename = 'sample-text-output'

	 

	device = torch.device('cuda')
	torch.cuda.empty_cache()

	category_lines , all_categories, start_letters = get_data(filename, all_letters)

	category_lines = shuffle_data(category_lines)

	split = int(np.floor(test_split * len(category_lines)))

	train_loader, test_loader, n_char = create_loader(category_lines, split, batch_size, all_letters, n_letters)

	lstm = StarTrekLSTM( n_char , n_layers , hidden_size , start_letters , n_letters , all_letters , device , dropout = 0.1)

	loss_func = nn.CrossEntropyLoss()

	optimizer=torch.optim.Adam(lstm.parameters(), lr=lr)


	train_losses_vis = []
	test_losses_vis = []
	test_acc_vis = []
	init_loss = 100000

	for epoch in range(1 , epochs + 1 ):
		print(f'Starting Epoch : {epoch}')

		train_loss = train(lstm, loss_func, device, train_loader, optimizer, epoch)
		test_loss, test_accuracy = test(lstm, loss_func, device, test_loader, epoch)

		if test_loss < init_loss:
			init_loss = test_loss
			torch.save(lstm.state_dict(), 'model_hw6.pt')
		test_acc_vis.append(test_accuracy)
		train_losses_vis.append(train_loss)
		test_losses_vis.append(test_loss)


		text = write_to_text(lstm, out_filename, epoch, 'epoch')

		print(text)


	return train_losses_vis, test_losses_vis, test_acc_vis



def shuffle_data(cat_lines):
	'''
	Function shuffles data

	Input :

	- cat_lines : Dict (with key 'st')

	Output:

	- category_lines : list ( shuffled list )
	'''

	category_lines = cat_lines['st']

	np.random.shuffle(category_lines)

	return category_lines



def run():

	'''
	Function to run the whole task for HW6
	'''

	filename = os.path.join('data', 'star_trek_transcripts_all_episodes_f.csv')
	all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-=|"

	n_letters = len(all_letters)

	train_losses_vis, test_losses_vis, test_acc_vis = custom_training(filename, all_letters, n_letters)


	plot_graph(train_losses_vis, 'TrainLoss')
	plot_graph(test_losses_vis, 'TestLoss') # test in this case is the test 
	plot_graph(test_acc_vis, 'TestAccuracy')

	print('Finished the task')





if __name__ == '__main__':
	run()
