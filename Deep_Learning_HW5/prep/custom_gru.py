'''
CUstom GRU 
'''

class CustomGRU(nn.Module):
	def __init__(self, vocab_size, n_layers, hidden_size, n_categories, device):
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.gru = nn.GRU(vocab_size, hidden_size, n_layers)
		self.fc = nn.Linear(hidden_size, n_categories)
		self.device = device
	def forward(self, input_item, lengths, states):
		hidden, cell = states[0], states[1]
		inputs = pack_padded_sequence(input_item, lengths)
		out, hidden = self.gru(inputs.to(device), hidden.to(device))
		out = self.fc(hidden[-1]) 
		return out, hidden

	def init_hidden(self, batch_size):
		# standard way - if you call .cuda() on the model it’ll return cuda tensors instead.
		hidden = Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size))
		cell =  Variable(next(self.parameters()).data.new(self.n_layers, batch_size, self.hidden_size))
		return hidden.zero_(), cell.zero_()