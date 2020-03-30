'''
Visualisation 
'''
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import os

def plot_graph(graph_list, n_type, name, g_type):

	'''
	Plotting the graph of loss/ accuracy

	- graph_list : list (list of values to be plotted ( in the format (x,y)))
	- n_type : string (network type: LSTM/ GRU)
	- name : string (desired file name)
	- g_type : string (graph label for the y axis)

	Outputs:
	None but saves graphs


	'''
	plt.plot(*zip(*graph_list), linewidth=2.0)
	plt.ylabel(g_type)
	plt.xlabel('Batch Size')

	directory = os.path.join('results',f'{n_type}_{name}_graph')
	
	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, f'{n_type}_{name}_graph.png'))
	plt.show()

def test_confusion_matrix(all_categories,confusion, batch_size, n_layers, h_size, n_type):
	'''
	Create a visualisation of the confusion matrix

	Inputs:
	- all_categories  : list (list of all categories in the dataset)
	- confusion : np array (Counting the number of correct prediction from the model)
	- batch_size : int (batch size)
	- n_layers : int (number of layers used)
	- h_size : int (hidden size)
	- n_type : string (type of network used: Lstm or gru)

	output:

	None ( but outputs and saves the confusion matrix )


	'''

	n_categories = len(all_categories)

	directory = os.path.join('results','test_confusion_matrix')
	
	for i in range(n_categories):
		confusion[i] = confusion[i] / confusion[i].sum()
	# Set up plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(confusion.numpy())
	fig.colorbar(cax)
	# Set up axes
	ax.set_xticklabels([''] + all_categories, rotation=90)
	ax.set_yticklabels([''] + all_categories)
	# Force label at every tick
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
	# sphinx_gallery_thumbnail_number = 2

	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, f'{n_type}_{n_layers}_{h_size}_{batch_size}_test_confusion_matrix.png'))
	plt.show()


def save_to_txt (text, n_type, n_layers, h_size, batch_size):
	directory = os.path.join('results',f'{n_type}')

	with open(f'results_{n_type}_{n_layers}_{h_size}_{batch_size}.txt', 'wb') as f:
		f.write(text)
		f.close()

