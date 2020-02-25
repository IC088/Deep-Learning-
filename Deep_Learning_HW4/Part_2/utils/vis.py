import matplotlib.pyplot as plt
import numpy as np

def visualise_train_val_loss( train_losses , val_losses, num_epochs , output_file):
	plt.figure(figsize=(12,5))
	epoch_list = np.arange(1, num_epochs+1)
	plt.xticks(epoch_list)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(epoch_list, train_losses, label="Training loss")
	plt.plot(epoch_list, val_losses, label="Validation loss")
	plt.legend(loc='upper right')

	plt.savefig(output_file)
	plt.show()