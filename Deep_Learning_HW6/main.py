'''
Deep Learning HW6

main file

Ivan Christian
'''

import os
import string
import csv



from loader.custom_lstm import CustomLSTM

import torch






def run():

	'''
	Function to run the whole task
	'''

	filename = os.path.join('data', 'star_trek_transcripts_all_episodes_f.csv')

	category_lines , all_categories = get_data(filename)

	n_categories = len(all_categories)
	lr = 0.1
	n_layers = 3
	h_size = 300
	dropout = 0.1

	device = torch.device('cuda')
	torch.cuda.empty_cache()





if __name__ == '__main__':
	run()
