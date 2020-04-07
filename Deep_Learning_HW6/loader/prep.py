'''
Data Preparation

Creating Dataset

DLHW6
'''

from io import StringIO

import random
import csv
import re

import torch

from torch.utils.data import Dataset



def unicodeToAscii(s, all_letters):
		return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

def read_lines(filename, all_letters):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line, all_letters) for line in lines]


def get_data(filename, all_letters):
	'''
	Function gets data to be filtered from the actual dataset (given by Prof)

	Inputs: 
	- filename: string ( file name for the star trek file in the dataset )

	Outputs:
	- category_lines : dict ( Conversation )
	- all_categories : list ( contains 'st' category )
	- start_letters : list (contains the starting letters)

	'''
	category_lines = {}
	all_categories = ['st']
	start_letters = []
	category_lines['st']=[]
	filterwords=['NEXTEPISODE']

	with open(filename, newline='') as f:
		io = StringIO(re.sub(r',[^a-zA-Z -]', '\t', f.read()))

		#Cleaning out the dataset using regex
		io = StringIO(re.sub(r', ', 'jkijkijki', io.getvalue())) #Replace the necessary commas with random string that will never occur in converstation
		io = StringIO(re.sub(r',', '\t', io.getvalue())) #get delimiter
		io = StringIO(re.sub(r'jkijkijki', ', ', io.getvalue())) # Replace back with original text

		reader = csv.reader(io, delimiter='\t', quotechar='"')
		for row in reader:
			for el in row:
				if (el not in filterwords) and (len(el)>1):
					v = re.sub(r'[;\"=/]', '', el)
					v += '|'
					category_lines['st'].append(v)
					start_letters.append(v[0])
	n_categories = len(all_categories)
	return category_lines , all_categories, start_letters

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


class STDataset(Dataset):

	def __init__(self, data, all_letters, n_letters):
		'''
		Initialise the 
		'''
		self.char_list = [[letterToIndex(char, all_letters) for char in line] for line in data]
		self.tensor_list = [lineToTensor(line,all_letters, n_letters) for line in data]

	def __len__(self):
		return len(self.tensor_list)
	
	def __getitem__(self, idx):
		return self.tensor_list[idx], self.char_list[idx]

	def getInputSize(self):
		return self.tensor_list[0].size()[-1]



