'''
Ivan Christian

Deep Learning HW5 Coding

LSTM and GRU
'''
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import os
from random import shuffle
import numpy as np

from prep.data_preprocessing import get_categories, lineToTensor, category_from_output
from prep.custom_lstm import CustomLSTM
from prep.custom_gru import CustomGRU

from vis.vis import test_confusion_matrix, plot_graph_train, plot_graph_test, save_to_txt


def pad_batch(batch,device):

	'''
	function to get the proper batch

	Inputs:
	- batch : list (batches)
	- device : string (cuda)

	Output:
	- list of the padded sequence and batches this should have the shape ready for LSTM analysis
	'''
	batch.sort(reverse = True, key = lambda x: x['length'])
	batch_names = [x['name_tensor'] for x in batch]
	batch_categories = [x['category_tensor'] for x in batch]
	batch_lengths = [x['length'] for x in batch]
	return [pad_sequence(batch_names).to(device), torch.stack(batch_categories).type(torch.LongTensor).to(device), torch.tensor(batch_lengths).to(device)]

def get_batches(category_lines, n_categories, all_categories, n_letters, all_letters, batch_size,device):
	'''
	Function is to create batches given the dataset and desired batch size

	Inputs:
	- category_lines: 
	- n_categories: int (number of categories)
	- all_categories: list (available categories)
	- n_letters: int (number of letters)
	- all_letters: string (all letters: abcdefghijklmnopqrstuvwxyz)
	- batch_size: int ( batch size)
	- device: string (cuda)

	Output:
	- batches : list (batches for train, test, val)
	'''
	batch_count = 0
	batches = []
	batch = []
	for data in category_lines:
		batch_count += 1
		sample = {}
		sample['name'] = data[0]
		sample['category'] = data[1]
		sample['length'] = data[2]
		sample['name_tensor'] = lineToTensor(data[0], n_letters, all_letters)
		sample['category_tensor'] = torch.tensor(all_categories.index(data[1]))
		batch.append(sample)
		if batch_count == batch_size:
			padded_batch = pad_batch(batch,device)
			batches.append(padded_batch)
			batch_count, batch = 0, []
	return batches



def custom_training_lstm(all_categories , batch_size, learning_rate, n_hidden, n_layers, rnn,train_set,val_set, test_set,num_epochs=5):
	'''
	Function to start the training process using lstm


	Inputs: 
	- all_categories : list (all categories available)
	- batch_size : int ( number of batches )
	- learning_rate : float ( learning rate )
	- n_hidden : int ( hidden size )
	- n_layers : int ( number of layers )
	- rnn : nn.Module class ( LSTM or GRU )
	- train_set : list ( train set according to batches )
	- val_set : list ( validation set according to batches )
	- test_set : list ( test set according to batches )
	- num_epochs : int ( number of epochs defaulted to 8 )

	Outputs :
	- train_losses_vis: list (train losses)
	- val_losses_vis: list (val losses)
	- val_accuracies: list (val accuracy list)
	- test_loss: float (average test loss)
	- test_accuracy: float (average test accuracy)
	- confusion: np array  (count the number of right predictions for visualisation.)
	'''

	n_categories = len(all_categories)
	def train(rnn, optimizer, epoch,loss_func):
		'''
		Inputs:
		- rnn : class (Custom class from the custom_gru)
		- optimizer : optimizer (optimiser, in this case specified to SGD) 
		- loss_func : function (loss function ,in this case )

		Outputs :
		- train_losses[0] : float (average loss)
		'''
		rnn.train()
		correct = 0
		running_loss = 0

		train_losses = []

		for idx, data in enumerate(train_set):
			optimizer.zero_grad()

			packed_names, categories, lengths = data[0], data[1], data[2]

			batch_size = categories.shape[0]
			hidden, cell = rnn.init_hidden(batch_size)
			output, _ = rnn(packed_names.data, lengths, (hidden, cell))
			loss = loss_func(output, categories)
			loss.backward()
			optimizer.step()
			running_loss += loss
			_, pred = torch.max(output, dim=1)
			correct += torch.sum(pred == categories)

		average_loss = running_loss / len(train_set)
		accuracy = correct.item() / (len(train_set) * batch_size) * 100.
		train_losses.append(average_loss.item())
		print(f'Training Epoch {epoch}\t Average Loss: {average_loss}\t Accuracy: {correct}/{len(train_set) * batch_size} ({np.round(accuracy, 4)}%)')
		return train_losses[0]
	def validate(rnn, epoch, loss_func):
		'''
		Function to start validation process as in to calculate the val loss and val accuracy

		Inputs: 
		- rnn : class (Custom class from custom_gru)
		- epoch : int (current number of epoch)
		- loss_func : loss function ( CrossEntropyLoss )

		Outputs : 
		- val_losses : list ( validation losses list )
		- val_Acc : list (Validation accuracy list)
		'''
		rnn.eval()
		correct = 0
		running_loss = 0

		val_losses = []
		val_acc = []

		with torch.no_grad():
			for idx, data in enumerate(val_set):
				packed_names, categories, lengths = data[0], data[1], data[2]
				batch_size = categories.shape[0]
				hidden, cell = rnn.init_hidden(batch_size)
				output, _ = rnn(packed_names.data, lengths, (hidden, cell))
				loss = loss_func(output, categories)
				running_loss += loss
				_, pred = torch.max(output, dim=1)
				correct += torch.sum(pred == categories)

		average_loss = running_loss / len(val_set)
		accuracy = correct.item() / (len(val_set) * batch_size) * 100.
		val_losses.append(average_loss.item())
		val_acc.append(accuracy)
		print(f'Validation Epoch {epoch}\t Average Loss: {average_loss}\t Accuracy: {correct}/{len(val_set) * batch_size} ({np.round(accuracy, 4)}%)')
		return val_losses[0], val_acc[0]

	def test(rnn, loss_func, n_categories):

		'''
		Function to start the test set

		Inputs:
		- rnn: class (Custom LSTM class defined in custom_lstm)
		- loss_func : loss function ( loss function to calc the loss, in this case is the CrossEntropyLoss)
		- n_categories : int ( number of categories which is 18)

		Output:
		- average_loss : float (average test loss)
		- accracy : float (Average test accuracy for the model)
		- confusion : tensor ( confusion matrix for visualisation)
		'''
		rnn.eval()

		correct = 0
		running_loss = 0

		confusion = torch.zeros(n_categories, n_categories)

		with torch.no_grad():
			for idx, data in enumerate(test_set):
				packed_names, categories, lengths = data[0], data[1], data[2]
				batch_size = categories.shape[0]
				hidden, cell = rnn.init_hidden(batch_size)
				output, _ = rnn(packed_names.data, lengths, (hidden, cell))
				loss = loss_func(output, categories)
				running_loss += loss
				_, pred = torch.max(output, dim=1)
				correct += torch.sum(pred == categories)

				g, _ = category_from_output(output, all_categories)
				c = categories.tolist()

				for i in range(len(g)):
					confusion[g[i], c[i]] += 1

		average_loss = running_loss / len(test_set)
		accuracy = correct.item() / (len(test_set) * batch_size) * 100

		print(f'Test Average Loss: {average_loss}\t Accuracy: {correct}/{len(test_set) * batch_size} ({np.round(accuracy, 4)}%)')

		return average_loss, accuracy, confusion




	loss_func = nn.CrossEntropyLoss()

	optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

	train_losses_vis = []
	val_accuracies = []
	val_losses_vis = []

	for epoch in range(1, num_epochs + 1):
		train_loss = train(rnn, optimizer, epoch, loss_func)

		val_loss, val_acc = validate(rnn, epoch, loss_func)
		train_losses_vis.append(train_loss)
		val_losses_vis.append(val_loss)

		val_accuracies.append(val_acc)

	test_loss, test_accuracy, confusion = test(rnn, loss_func, n_categories)

	return train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion


def custom_training_gru(all_categories , batch_size, learning_rate, n_hidden, n_layers, rnn,train_set,val_set, test_set,num_epochs=5):
	'''
	Function to train using gru

	Inputs: 
	- all_categories : list (all categories available)
	- batch_size : int ( number of batches )
	- learning_rate : float ( learning rate )
	- n_hidden : int ( hidden size )
	- n_layers : int ( number of layers )
	- rnn : nn.Module class ( LSTM or GRU )
	- train_set : list ( train set according to batches )
	- val_set : list ( validation set according to batches )
	- test_set : list ( test set according to batches )
	- num_epochs : int ( number of epochs defaulted to 8 )

	Outputs :
	- train_losses_vis: list (train losses)
	- val_losses_vis: list (val losses)
	- val_accuracies: list (val accuracy list)
	- test_loss: float (average test loss)
	- test_accuracy: float (average test accuracy)
	- confusion: np array  (count the number of right predictions for visualisation.)
	'''


	n_categories = len(all_categories)
	def train(rnn, optimizer, epoch,loss_func):

		'''
		Function to start the training process

		Inputs:
		- rnn : class (Custom class from the custom_gru)
		- optimizer : optimizer (optimiser, in this case specified to SGD) 
		- loss_func : function (loss function ,in this case )

		Outputs :
		- train_losses[0] : float (average loss)
		'''
		rnn.train()
		correct = 0
		running_loss = 0

		train_losses = []

		for idx, data in enumerate(train_set):
			optimizer.zero_grad()
	
			packed_names, categories, lengths = data[0], data[1], data[2]

			batch_size = categories.shape[0]
			hidden = rnn.init_hidden(batch_size)
			output, _ = rnn(packed_names.data, lengths, hidden)
			loss = loss_func(output, categories)
			loss.backward()
			optimizer.step()
			running_loss += loss
			_, pred = torch.max(output, dim=1)
			correct += torch.sum(pred == categories)

		average_loss = running_loss / len(train_set)
		accuracy = correct.item() / (len(train_set) * batch_size) * 100.
		train_losses.append(average_loss.item())
		print(f'Training Epoch {epoch}\t Average Loss: {average_loss}\t Accuracy: {correct}/{len(train_set) * batch_size} ({np.round(accuracy, 4)}%)')
		return train_losses[0]
	def validate(rnn, epoch, loss_func):
		'''
		Function to start validation process as in to calculate the val loss and val accuracy

		Inputs: 
		- rnn : class (Custom class from custom_gru)
		- epoch : int (current number of epoch)
		- loss_func : loss function ( CrossEntropyLoss )

		Outputs : 
		- val_losses : list ( validation losses list )
		- val_Acc : list (Validation accuracy list)
		'''
		rnn.eval()
		correct = 0
		running_loss = 0

		val_losses = []
		val_acc = []

		with torch.no_grad():
			for idx, data in enumerate(val_set):
				packed_names, categories, lengths = data[0], data[1], data[2]
				batch_size = categories.shape[0]
				hidden = rnn.init_hidden(batch_size)
				output, _ = rnn(packed_names.data, lengths, hidden)
				loss = loss_func(output, categories)
				running_loss += loss
				_, pred = torch.max(output, dim=1)
				correct += torch.sum(pred == categories)

		average_loss = running_loss / len(val_set)
		accuracy = correct.item() / (len(val_set) * batch_size) * 100.
		val_losses.append(average_loss.item())
		val_acc.append(accuracy)
		print(f'Validation Epoch {epoch}\t Average Loss: {average_loss}\t Accuracy: {correct}/{len(val_set) * batch_size} ({np.round(accuracy, 4)}%)')
		return val_losses[0], val_acc[0]

	def test(rnn, loss_func, n_categories):

		'''
		Function to start the test set

		Inputs:
		- rnn: class (Custom GRU class defined in custom_gru)
		- loss_func : loss function ( loss function to calc the loss, in this case is the CrossEntropyLoss)
		- n_categories : int ( number of categories which is 18)

		Output:
		- average_loss : float (average test loss)
		- accracy : float (Average test accuracy for the model)
		- confusion : tensor ( confusion matrix for visualisation)
		'''
		rnn.eval()

		correct = 0
		running_loss = 0

		confusion = torch.zeros(n_categories, n_categories)

		with torch.no_grad():
			for idx, data in enumerate(test_set):
				packed_names, categories, lengths = data[0], data[1], data[2]
				batch_size = categories.shape[0]
				hidden = rnn.init_hidden(batch_size)
				output, _ = rnn(packed_names.data, lengths, hidden)
				loss = loss_func(output, categories)
				running_loss += loss
				_, pred = torch.max(output, dim=1)
				correct += torch.sum(pred == categories)

				g, _ = category_from_output(output, all_categories)
				c = categories.tolist()

				for i in range(len(g)):
					confusion[g[i], c[i]] += 1

		average_loss = running_loss / len(test_set)
		accuracy = correct.item() / (len(test_set) * batch_size) * 100

		print(f'Test Average Loss: {average_loss}\t Accuracy: {correct}/{len(test_set) * batch_size} ({np.round(accuracy, 4)}%)')

		return average_loss, accuracy, confusion

	loss_func = nn.CrossEntropyLoss()

	optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

	train_losses_vis = []
	val_accuracies = []
	val_losses_vis = []

	for epoch in range(1, num_epochs + 1):
		
		train_loss = train(rnn, optimizer, epoch, loss_func)

		val_loss, val_acc = validate(rnn, epoch, loss_func)
		train_losses_vis.append(train_loss)
		val_losses_vis.append(val_loss)

		val_accuracies.append(val_acc)

	test_loss, test_accuracy, confusion = test(rnn, loss_func, n_categories)

	return train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion



def run():

	'''
	Main function to run the tasks 
	'''
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	path = os.path.join('data', 'names', '*.txt')
	all_letters = string.ascii_letters + " .,;'"

	category_lines, all_categories = get_categories(path,all_letters)

	shuffle(category_lines) #Further shuffle the 

	n_letters = len(all_letters)
	n_categories = len(all_categories)
	lr = 0.1
	torch.cuda.empty_cache()

	# TASK 1

	# LSTM with different n_layers and different hidden_sizes, batch size = 1.

	batch_size = 1

	n_layers_list = [1,2]

	hidden_sizes = [200 , 500]

	# 0.8 - 0.1 - 0.1 train val test split
	# Total Length of category_lines = 20047
	# floor(0.8*20047) = 16059

	train_set = get_batches(category_lines[:16058], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	val_set = get_batches(category_lines[16058:18066], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	test_set = get_batches(category_lines[18066:], n_categories, all_categories, n_letters, all_letters, batch_size, device)

	train_history = []
	val_history = []
	test_history = []

	n_type ='lstm'

	for n_layers in n_layers_list:
		for h_size in hidden_sizes:

			print(f'Doing LSTM with {n_layers} layers and {h_size} hidden sizes. Batch_size = {batch_size}')

			lstm = CustomLSTM(n_letters, n_layers, h_size, n_categories, device).to(device)

			train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion = custom_training_lstm(all_categories,batch_size,lr, h_size, n_layers, lstm, train_set,val_set, test_set)
			

			saved_file = f'LSTM\nTrain Loss : {train_losses_vis}\n Val Loss : {val_losses_vis}\n Val Accuracy: {val_accuracies}\n Test Loss : {test_loss}\n Test Accuracy: {test_accuracy}'

			print(saved_file)

			save_to_txt(saved_file, n_type, n_layers, h_size, batch_size, 1)

			train_history.append(train_losses_vis)
			val_history.append([val_losses_vis, val_accuracies])
			test_history.append([test_loss.item(), test_accuracy, (batch_size, n_layers, h_size)])

			test_confusion_matrix(all_categories,confusion, batch_size, n_layers, h_size, n_type, 1)
			torch.cuda.empty_cache()

	# TASK 1: GRU

	# GRU with hidden size 200 and layers = 1, batch size 1

	n_layers = 1
	h_size = 200
	n_type = 'gru'

	gru = CustomGRU(n_letters, n_layers, h_size, n_categories, device).to(device)
	train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion = custom_training_gru(all_categories,batch_size,lr, h_size, n_layers, gru, train_set,val_set, test_set)

	test_confusion_matrix(all_categories,confusion, batch_size, n_layers, h_size, n_type, 1)
	torch.cuda.empty_cache()

	saved_file = f'GRU \nTrain Loss : {train_losses_vis}\n Val Loss : {val_losses_vis}\n Val Accuracy: {val_accuracies}\n Test Loss : {test_loss}\n Test Accuracy: {test_accuracy}'

	print(saved_file)
	save_to_txt(saved_file, n_type, n_layers, h_size, batch_size, 1)


	# TASK 2 : LSTM with different batch sizes make graph based on the test loss, test accuracy, and train loss
	
	torch.cuda.empty_cache()

	batch_size_list = [(1,5),(10,5),(50,5)]

	print(f'STARTING TASK 2 for batch sizes 1,10,50')
	h_size = 500
	n_layers = 2
	n_type = 'lstm'
	train_loss_history = []
	test_loss_history = []
	test_acc_history = []
	val_loss_history = []
	val_acc_history = []

	for b_size, epoch in batch_size_list:
		print(f'Doing LSTM with {n_layers} layers and {h_size} hidden sizes. Batch_size = {b_size}')


		train_set = get_batches(category_lines[:16058], n_categories, all_categories, n_letters, all_letters, b_size,device)
		val_set = get_batches(category_lines[16058:18066], n_categories, all_categories, n_letters, all_letters, b_size,device)
		test_set = get_batches(category_lines[18066:], n_categories, all_categories, n_letters, all_letters, b_size, device)
		lstm = CustomLSTM(n_letters, n_layers, h_size, n_categories, device).to(device)

		train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion = custom_training_lstm(all_categories,b_size,lr, h_size, n_layers, lstm, train_set,val_set, test_set, num_epochs=epoch)
		
		saved_file = f'Train Loss : {train_losses_vis}\nVal Loss : {val_losses_vis}\nVal Accuracy: {val_accuracies}\nTest Loss : {test_loss}\nTest Accuracy: {test_accuracy}'
		print('>'*30)
		print(saved_file)
		print('<'*30)

		save_to_txt(saved_file, n_type, n_layers, h_size, b_size, 2)

		torch.cuda.empty_cache()
		test_confusion_matrix(all_categories,confusion, b_size, n_layers, h_size, n_type, 2)
		

		for i in range(len(train_losses_vis)):
			train_loss_history.append((b_size, train_losses_vis[i]))
			val_loss_history.append((b_size, val_losses_vis[i]))
			val_acc_history.append((b_size, val_accuracies[i]))
		

		test_loss_history.append((b_size, float(test_loss)))
		test_acc_history.append((b_size, test_accuracy))
		

		torch.cuda.empty_cache()


	print('>'*30)
	print('LSTM')
	print(f'test_loss_history : {test_loss_history}')
	print(f'train_loss_history : {train_loss_history}')
	print(f'test_acc_history : {test_acc_history}')
	print('<'*30)
	
	plot_graph_test(test_acc_history, n_type, 'Graph of Test Accuracy against batch size', 'Test Accuracy')
	plot_graph_test(test_loss_history, n_type, 'Graph of Test Loss against batch size', 'Test Loss')
	plot_graph_train(train_loss_history, n_type, 'Graph of Train Loss against batch size', 'Train Loss')

if __name__ == '__main__':
	run()
