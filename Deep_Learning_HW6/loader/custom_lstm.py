'''
Ivan Christian

LSTM Custom
'''


import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import numpy as np


def letterToIndex(letter, all_letters):
	if all_letters.find(letter) == -1:
		print ("char %s not found in vocab" %letter) #find missing letters
	return all_letters.find(letter)

def indexToLetter(all_letters, idx):
	return all_letters[idx]

def lineToTensor(line, all_letters,n_letters):
	tensor = torch.zeros(len(line), n_letters)
	for li, letter in enumerate(line):
		tensor[li][letterToIndex(letter,all_letters)] = 1
	return tensor



class StarTrekLSTM(nn.Module):

	'''
	Custom LSTM modules based on the  tutorial 
	'''
	def __init__(self, n_chars, n_layers, hidden_size, start_letters, n_letters, all_letters, device, dropout = 0.1):
		super(StarTrekLSTM, self).__init__()

		'''
		Inputs:
		- n_chars : input_size
		- n_layers : ( 3 layers)
		- hidden_size : ( this will be 300 for this Star Trek task )
		'''
		self.n_chars = n_chars
		self.n_layers = n_layers
		self.hidden_size = hidden_size

		self.start_letters = start_letters
		self.n_letters = n_letters
		self.all_letters = all_letters

		self.lstm = nn.LSTM(n_chars, hidden_size, n_layers, dropout=dropout, batch_first=True).to(device)
		self.fc = nn.Linear(hidden_size, n_chars).to(device)
		self.device = device

	def forward(self, inp):
		num_batch, _, _ = inp.shape
		h = torch.zeros(self.n_layers, num_batch, self.hidden_size).to(self.device)
		c = torch.zeros(self.n_layers, num_batch, self.hidden_size).to(self.device)

		output, _ = self.lstm(inp, (h, c))
		out = self.fc((output.squeeze()))

		return out

	def sample(self, temp):
		'''
		Sampling based on the temperature
		'''

		self.eval() # set to eval mode

		curr_letter = np.random.choice(self.start_letters) #start with uppercase

		sentence = ''

		h = torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device)
		c = torch.zeros(self.n_layers, 1, self.hidden_size).to(self.device)

		with torch.no_grad():

			while curr_letter != '|':

				sentence += curr_letter
				t = torch.zeros(1, 1, self.n_letters).to(self.device)
				t[0][0][letterToIndex(curr_letter, self.all_letters)] = 1
				output, (h, c) = self.lstm(t, (h, c))
				output = self.fc(output.squeeze()).squeeze() 
				probs = torch.nn.functional.softmax(output/temp, dim=0).cpu().numpy() # Must be in CPU
				curr_idx = np.random.choice(np.arange(self.n_letters), p=probs) # Set letter with highest probability
				curr_letter = indexToLetter(self.all_letters, curr_idx)

		return sentence



