from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string


'''
Taken  from the tutorial
'''

import torch

import numpy as np



def unicodeToAscii(s, all_letters):
		return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, all_letters):
	return all_letters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line, n_letters, all_letters):
	tensor = torch.zeros(len(line), n_letters)
	for li, letter in enumerate(line):
		tensor[li][letterToIndex(letter, all_letters)] = 1
	return tensor

# Read a file and split into lines
def read_lines(filename, all_letters):
	lines = open(filename, encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line, all_letters) for line in lines]

def get_categories(path, all_letters):
	category_lines = []
	all_categories = []
	for filename in glob.glob(path):
		category = os.path.splitext(os.path.basename(filename))[0]
		all_categories.append(category)
		lines = read_lines(filename, all_letters)
		for line in lines:
			category_lines.append([line, category, len(line)])    
	np.random.shuffle(category_lines)
	return category_lines, all_categories

	
def category_from_output(output, all_categories):
	categories = []
	winners = []
	for row in range(output.shape[0]):
		top_n, top_i = output[row].topk(1)
		winner = top_i[0].item()
		categories.append(all_categories[winner])
		winners.append(winner)
	return winners, categories
