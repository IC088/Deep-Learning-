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

import numpy as np

from prep.data_preprocessing import get_categories, lineToTensor, category_from_output
from prep.custom_lstm import CustomLSTM
from prep.custom_gru import CustomGRU

from vis.vis import train_loss_graph, val_loss_graph, val_acc_graph, test_confusion_matrix


def pad_batch(batch,device):
	batch.sort(reverse = True, key = lambda x: x['length'])
	batch_names = [x['name_tensor'] for x in batch]
	batch_categories = [x['category_tensor'] for x in batch]
	batch_lengths = [x['length'] for x in batch]
	return [pad_sequence(batch_names).to(device), torch.stack(batch_categories).type(torch.LongTensor).to(device), torch.tensor(batch_lengths).to(device)]

def get_batches(category_lines, n_categories, all_categories, n_letters, all_letters, batch_size,device):
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



def custom_training_lstm(all_categories , batch_size, learning_rate, n_hidden, n_layers, rnn,train_set,val_set, test_set,num_epochs=10):
	n_categories = len(all_categories)
	def train(rnn, optimizer, epoch,loss_func):

		'''
		Function to start the training process
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
		return train_losses
	def validate(rnn, epoch, loss_func):
		'''
		Function to start validation process
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
		return val_losses, val_acc

	def test(rnn, loss_func, n_categories):

		'''
		Function to start the test set
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
		train_losses_vis = train(rnn, optimizer, epoch, loss_func)
		val_losses_vis, val_accuracies = validate(rnn, epoch, loss_func)

	test_loss, test_accuracy, confusion = test(rnn, loss_func, n_categories)

	return train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion


def custom_training_gru(all_categories , batch_size, learning_rate, n_hidden, n_layers, rnn,train_set,val_set, test_set,num_epochs=10):
	n_categories = len(all_categories)
	def train(rnn, optimizer, epoch,loss_func):

		'''
		Function to start the training process
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
		return train_losses
	def validate(rnn, epoch, loss_func):
		'''
		Function to start validation process
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
		return val_losses, val_acc

	def test(rnn, loss_func, n_categories):

		'''
		Function to start the test set
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
		train_losses_vis = train(rnn, optimizer, epoch, loss_func)
		val_losses_vis, val_accuracies = validate(rnn, epoch, loss_func)

	test_loss, test_accuracy, confusion = test(rnn, loss_func, n_categories)

	return train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion



def run():

	'''
	Main function to run the task 
	'''
	lr = 0.01

	batch_size = 1

	n_layers_list = [1,2]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	path = os.path.join('data', 'names', '*.txt')
	all_letters = string.ascii_letters + " .,;'"

	category_lines, all_categories = get_categories(path,all_letters)

	n_letters = len(all_letters)
	n_categories = len(all_categories)

	hidden_sizes = [200 , 500]


	# 0.7 - 0.1 - 0.2 train val test split

	train_set = get_batches(category_lines[:14051], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	val_set = get_batches(category_lines[14051:16058], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	test_set = get_batches(category_lines[16058:], n_categories, all_categories, n_letters, all_letters, batch_size, device)


	#TASK 1

	# LSTM with different n_layers and different hidden_sizes, batch size = 1.

	# n_type ='lstm'

	# train_history = []
	# test_history = []
	# val_history = []

	# for n_layers in n_layers_list:
	# 	for h_size in hidden_sizes:

	# 		print(f'Doing LSTM with {n_layers} layers and {h_size} hidden sizes. Batch_size = {batch_size}')

	# 		lstm = CustomLSTM(n_letters, n_layers, h_size, n_categories, device).to(device)

	# 		train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion = custom_training_lstm(all_categories,batch_size,lr, h_size, n_layers, lstm, train_set,val_set, test_set)

	# 		train_history.append(train_losses_vis)
	# 		val_history.append([val_losses_vis, val_accuracies])
	# 		test_history.append([test_loss.item(), test_accuracy, (batch_size, n_layers, h_size)])

	# 		test_confusion_matrix(all_categories,confusion, batch_size, n_layers, h_size, n_type)

	# # train_loss_graph(train_history,batch_size, n_layers, h_size, n_type)
	# # val_loss_graph(val_history[0],batch_size, n_layers, h_size, n_type)
	# # val_acc_graph(val_history[1],batch_size, n_layers, h_size, n_type)



	# TASK 1: GRU

	# GRU with hidden size 200 and layers = 1, batch size 1

	n_layers = 1
	h_size = 200
	n_type = 'gru'

	gru = CustomGRU(n_letters, n_layers, h_size, n_categories, device).to(device)
	train_losses_vis, val_losses_vis, val_accuracies, test_loss, test_accuracy, confusion = custom_training_gru(all_categories,batch_size,lr, h_size, n_layers, gru, train_set,val_set, test_set)


	print('Train Loss')
	print(f'Train Loss : {train_losses_vis}\n Val Loss : {val_losses_vis}\n Val Accuracy: {val_accuracies}\n Test Loss : {test_loss}\n Test Accuracy: {test_accuracy}')


	# TASK 2 : LSTM with different batch sizes

	batch_size_list = [10,50]

	for b_size in batch_size_list:
		train_set = get_batches(category_lines[:14051], n_categories, all_categories, n_letters, all_letters, b_size,device)
		val_set = get_batches(category_lines[14051:16058], n_categories, all_categories, n_letters, all_letters, b_size,device)
		test_set = get_batches(category_lines[16058:], n_categories, all_categories, n_letters, all_letters, b_size, device)




	# TASK 2 : LSTM 

if __name__ == '__main__':
	run()
