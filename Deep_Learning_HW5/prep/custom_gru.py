'''
CUstom GRU 
'''

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class CustomGRU(nn.Module):
	def __init__(self, vocab_size, n_layers, hidden_size, n_categories, device):

		super(CustomGRU, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.gru = nn.GRU(vocab_size, hidden_size, n_layers)
		self.fc = nn.Linear(hidden_size, n_categories)

		self.softmax = nn.LogSoftmax(dim=1)
		self.device = device
	def forward(self, input_item, lengths, states):
		hidden = states
		inputs = pack_padded_sequence(input_item, lengths)
		out, hidden = self.gru(inputs.to(self.device), hidden.to(self.device))
		out = self.fc(hidden[-1]) 
		out = self.softmax(out)
		return out, hidden

	def init_hidden(self, batch_size):
		# standard way - if you call .cuda() on the model it’ll return cuda tensors instead.
		hidden = Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size)).to(self.device)
		return hidden.zero_()