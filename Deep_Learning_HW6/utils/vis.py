'''
Utilities for visualisation and other stuff

DLHW6
'''
import os

import matplotlib.pyplot as plt



def write_to_text(model, filename, epoch, train_or_epoch ,temp = 0.5):
	'''
	Function to save the sample temperatures to a text file

	Input:
	- model : custom model object ( This is the LSTM model which contains the function to extract samples)
	- filename : string ( name of the file to be saved)
	- epoch : int ( epoch number )
	- train_or_epoch : string (accepts 'train' or 'epoch')
	- temp : float ( a temperature value )

	Output:
	- String to indicate the completion 
	'''

	assert (train_or_epoch == 'train' or train_or_epoch == 'epoch'),'Mode Not Valid'
	directory = os.path.join('results', 'sample_text')

	if not os.path.exists(directory):
		os.makedirs(directory)


	full_path = os.path.join(directory, f'{filename}-{epoch}.txt')
	with open(full_path , 'w+') as f:

		for i in range(5):
			text = model.sample(temp)

			f.write(f'{text} \n')
		f.close()

	if train_or_epoch == 'epoch':
		return f'Finished epoch {epoch}'
	else:
		return f'Finished train samples {epoch}'


def plot_graph(data_list, filename):
	'''
	Function to plot graph based on the list of values

	Input:
	- data_list : list (list of values of either training loss, val loss, val accuracy)
	- filename : string ( name of the intended file )

	Output:
	 Returns None but saves thee graph to results folder
	'''
	directory = os.path.join('results', 'graphs')

	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.plot(data_list, label=filename)
	plt.legend(loc='upper left')
	plt.savefig(os.path.join(directory , f'{filename}.png'))
	plt.clf()



