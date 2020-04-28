
import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import math


# # Encoder, Decoder, Generator, Discriminator


# Encoder Decoder for the basic Generator
class EncoderCNN(nn.Module):
	def __init__(self, embed_size):
		super(EncoderCNN, self).__init__()
		resnet = models.resnet50(pretrained=True)
		for param in resnet.parameters():
			param.requires_grad_(False)
		
		modules = list(resnet.children())[:-1]
		self.resnet = nn.Sequential(*modules)
		self.embed = nn.Linear(resnet.fc.in_features, embed_size)
		self.train_params = list(self.embed.parameters())

	def forward(self, images):
		features = self.resnet(images)
		features = features.view(features.size(0), -1)
		features = self.embed(features)
		return features
	

class DecoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
		super().__init__()
		self.embedding_layer = nn.Embedding(vocab_size, embed_size)
		
		self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
							num_layers = num_layers, batch_first = True)
		
		self.linear = nn.Linear(hidden_size, vocab_size)
		self.train_params = list(self.parameters())
	
	def forward(self, features, captions):
		captions = captions[:, :-1]
		embed = self.embedding_layer(captions)
		embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
		lstm_outputs, _ = self.lstm(embed)
		out = self.linear(lstm_outputs)
		
		return out

	def sample(self, inputs, states=None, max_len=20):
		" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
		output_sentence = []
		for i in range(max_len):
			lstm_outputs, states = self.lstm(inputs, states)
			lstm_outputs = lstm_outputs.squeeze(1)
			out = self.linear(lstm_outputs)
			last_pick = out.max(1)[1]
			output_sentence.append(last_pick.item())
			inputs = self.embedding_layer(last_pick).unsqueeze(1)
		
		return output_sentence
	
	def beam_sample(self, inputs, states=None, max_len=20, k=1):
		" accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
		possible_seq = [(1, inputs, states)]
		for i in range(max_len):
			to_pick = []
			for probs,seq,states in possible_seq:
				inputs = self.embedding_layer(seq[-1])
				lstm_outputs, states = self.lstm(inputs, states)
				out = self.linear(lstm_outputs).squeeze(0)
				sorted_out, indices = torch.sort(out, 1)
				
				for j in range(k):
					to_pick.append((probs + nn.functional.log_softmax(sorted_out[i]), inputs + [indices[i]], states) )
				 
			to_pick.sort(reverse=True)
			possible_seq = to_pick[:k]
			
		return to_pick[0]


class EncoderRNN(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, context_size, num_layers=1):
		super().__init__()
		#self.embedding_layer = nn.Embedding(vocab_size, embed_size)
		self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,
							num_layers = num_layers, batch_first = True)
		self.linear = nn.Linear(hidden_size, context_size)
		self.train_params = list(self.parameters())
	
	def forward(self, captions):
		lstm_outputs, _ = self.lstm(captions)
		out = self.linear(lstm_outputs[:,-1,:].squeeze(1))
		return out

	
class Generator(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, context_size, num_layers=1):
		super(Generator, self).__init__()
		self.cnn = EncoderCNN(context_size)
		self.rnn = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers)
		self.train_params = self.cnn.train_params + self.rnn.train_params 

	def forward(self, images, captions):
		features = self.cnn(images)
		output = self.rnn(features, captions)
		return output, features


class Discriminator(nn.Module):
	def __init__(self, embed_size, hidden_size, vocab_size, context_size, num_layers=1):
		super(Discriminator, self).__init__()
		self.embed = nn.Linear(vocab_size, embed_size)
		self.rnn = EncoderRNN(embed_size, hidden_size, vocab_size, context_size, num_layers=num_layers)
		#self.crit = nn.CosineSimilarity(dim=0, eps=1e-6)
		self.train_params = list(self.embed.parameters()) + self.rnn.train_params
		

	def forward(self, image_feat, captions):
		bs = image_feat.size(0)
		embed = self.embed(captions)
		cap_feat = self.rnn(embed)
			
		return cap_feat

