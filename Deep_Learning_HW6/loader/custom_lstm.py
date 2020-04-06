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
	def __init__(self, n_chars, n_layers, hidden_size, device, dropout = 0.1):
		super(CustomLSTM, self).__init__()

		'''
		Inputs:
		- n_chars : 
		- n_layers : ( 3 layers)
		- hidden_size : ( this will be 300 for this Star Trek task )
		'''
		self.n_chars = n_chars
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(n_chars, hidden_size, n_layers, dropout=dropout, batch_first=True)
		self.fc = nn.Linear(hidden_size, n_chars)
		self.device = device

		
	def forward(self, sentence, lengths, hc):
		'''
		Forward in the LSTM
		'''
		batch_size, sequence_length = sentence.size()[:-1]

		sentence = pack_padded_sequence(sentence, lengths,batch_first=True)
		lstm_out, hc = self.lstm(sentence.to(self.device), hc.to(self.device))
		lstm_out = lstm_out.contigous() # Otherwise errors
		lstm_out = lstm_out.view(-1, lstm_out.shape[2])
		out = self.fc(lstm_out)
		out = out.view(batch_size, sequence_length, self.n_chars)
		return out, hc

	def init_hidden(self, batch_size):
		'''
		Initialisation
		'''

		weight = next(self.parameters()).data


		hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device)
		cell =  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device)
		return (hidden,cell)


