'''
Ivan Christian

LSTM Custom
'''


import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class CustomLSTM(nn.Module):

	'''
	Custom LSTM modules based on the  tutorial 
	'''
	def __init__(self, vocab_size, n_layers, hidden_size, n_categories, device):
		super(CustomLSTM, self).__init__()

		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(vocab_size, hidden_size, n_layers)
		self.fc = nn.Linear(hidden_size, n_categories)
		self.device = device

		
	def forward(self, input_item, lengths, states):
		hidden, cell = states[0].to(self.device), states[1].to(self.device)
		inputs = pack_padded_sequence(input_item, lengths)
		out, (hidden, cell) = self.lstm(inputs.to(self.device), (hidden.to(self.device), cell.to(self.device)))
		out = self.fc(hidden[-1]) 
		return out, (hidden, cell)

	def init_hidden(self, batch_size):
		hidden = Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size))
		cell =  Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size))
		return hidden.zero_(), cell.zero_()


