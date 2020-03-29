'''
Visualisation 
'''
import matplotlib.pyplot as plt

import os



def train_loss_graph(train_losses_vis,batch_size, n_layers, h_size, n_type):
	plt.plot(train_losses_vis)

	directory = os.path.join('results','train_loss_graph')
	
	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, f'{n_type}_{n_layers}_{h_size}_{batch_size}_train_loss_graph.png'))
	plt.show()


def val_loss_graph(val_loss_vis,batch_size, n_layers, h_size, n_type):
	plt.plot(val_losses_vis)

	directory = os.path.join('results','val_loss_graph')
	
	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, f'{n_type}_{n_layers}_{h_size}_{batch_size}_val_loss_graph.png'))
	plt.show()
def val_acc_graph(val_acc_vis,batch_size, n_layers, h_size, n_type):
	plt.plot(val_acc_vis)

	directory = os.path.join('results',f'{n_type}_{n_layers}_{h_size}_{batch_size}_val_acc_graph')
	
	if not os.path.exists(directory):
		os.makedirs(directory)

	plt.savefig(os.path.join(directory, f'{n_type}_{n_layers}_{h_size}_{batch_size}_val_acc_graph.png'))
	plt.show()

def test_confusion_matrix(all_categories,confusion, batch_size, n_layers, h_size, n_type):

	n_categories = len(all_categories)

	directory = os.path.join('results','test_confusion_matrix')
	
	# Normalize by dividing every row by its sum
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