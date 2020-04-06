'''
Data Preparation

DLHW6
'''

import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# Random item from a list
def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair(category_lines ,all_categories):
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	return category, line

def get_data(filename):
	'''
	Function gets data to be filtered from the actual dataset

	Inputs: 
	- filename: string ( file name for the star trek file in the dataset )

	Outputs:
	- category_lines : dict ( Conversation )
	- all_categories : list ( contains 'st' category )

	'''
	all_letters = string.ascii_letters + "0123456789 .,:!?'[]()/+-="
	category_lines = {}
	all_categories = ['st']
	category_lines['st']=[]
	filterwords=['NEXTEPISODE']

	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		for row in reader:
			for el in row:
				if (el not in filterwords) and (len(el)>1):
					v = el.strip().replace(';','').replace('\"','')
					category_lines['st'].append(v)
	
	return category_lines , all_categories




class DSet(Dataset):
