'''
Ivan Christian

Deep Learning HW5

RNN & LSTM
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

from prep.data_preprocessing import get_categories, lineToTensor
from prep.custom_lstm import CustomLSTM


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



def custom_training(batch_size, learning_rate, n_hidden, n_layers, rnn,train_set,val_set, test_set,num_epochs=5):
	def compute_loss_weights():
		clsses = np.arange(0,18) # n_categories
		d = [{}, {}, {}]
		for i in range(18):
			d[0][all_categories[i]] = 0
			d[1][all_categories[i]] = 0
			d[2][all_categories[i]] = 0

		sets = [train_set, val_set, test_set]
		for idx in range(len(sets)):
			# for variable length sets (x is batch idx)
			for x in range(len(sets[idx])):
				# for every class index in that batch
				for cls in sets[idx][x][1]:
					d[idx][all_categories[cls.item()]] += 1
		train_distribution = np.array(list((d[0].values())))
		minimum_frequency = np.min(train_distribution)
		
		# Here use either of the following. Remember to scale
		# the learning rate accordingly.
		# loss_weights = minimum_frequency / train_distribution
		# loss_weights = 1 / train_distribution
		loss_weights = 1 / train_distribution
		return torch.FloatTensor(loss_weights).to(device)


	def train(rnn, optimizer, epoch,loss_func):
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

	loss_func = nn.CrossEntropyLoss()

	optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

	train_losses_vis = []
	val_accuracies = []
	vall_losses_vis = []

	for epoch in range(1, num_epochs + 1):
		train_losses_vis = train(rnn, optimizer, epoch, loss_func)
		vall_losses_vis, val_accuracies = validate(rnn, epoch, loss_func)


def run():

	batch_size_list = [1,10,30]
	lr = 0.01

	batch_size = 1

	n_layers_list = [1,2]
	n_layers = 1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	path = os.path.join('data', 'names', '*.txt')
	all_letters = string.ascii_letters + " .,;'"

	category_lines, all_categories = get_categories(path,all_letters)
	n_letters = len(all_letters)
	n_categories = len(all_categories)

	hidden_size = 200


	# 0.7 - 0.1 - 0.2 train val test split

	train_set = get_batches(category_lines[:14051], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	val_set = get_batches(category_lines[14051:16058], n_categories, all_categories, n_letters, all_letters, batch_size,device)
	test_set = get_batches(category_lines[16058:], n_categories, all_categories, n_letters, all_letters, batch_size, device)


	lstm_1 = CustomLSTM(n_letters, n_layers, hidden_size, n_categories, device).to(device)


	custom_training(batch_size,lr, hidden_size, n_layers, lstm_1, train_set,val_set, test_set) #lstm with 1 layer




if __name__ == '__main__':
	run()
