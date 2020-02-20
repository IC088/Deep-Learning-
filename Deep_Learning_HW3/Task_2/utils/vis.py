import numpy as np
import matplotlib.pyplot as plt

def plot_training_validation_loss(num_epochs, train_losses, test_losses):
    epoch_list = np.arange(1, num_epochs+1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, train_losses, label="Training loss")
    plt.plot(epoch_list, test_losses, label="Validation loss")
    plt.legend(loc='upper right')
    plt.show()