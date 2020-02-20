import numpy as np
import matplotlib.pyplot as plt

def plot_training_validation_loss(num_epochs, train_losses, test_losses, ty='val'):
    if ty == 'val':

        epoch_list = np.arange(1, num_epochs+1)
        plt.xticks(epoch_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_list, train_losses, label="Training loss")
        plt.plot(epoch_list, test_losses, label="Validation loss")
        plt.plot(epoch_list, test_losses, label = "Test Loss")
        plt.legend(loc='upper right')
        plt.savefig('.\\loss-graph.png')
        plt.show()
    else:
        epoch_list = np.arange(1, num_epochs+1)
        plt.xticks(epoch_list)
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.plot(epoch_list, train_losses, label="Training Acc")
        plt.plot(epoch_list, test_losses, label="Validation Acc")
        plt.plot(epoch_list, test_losses, label = "Test Acc")
        plt.legend(loc='upper right')

        plt.savefig('.\\Acc-graph.png')
        plt.show()


def plot_classwise_accuracies(classwise_accuracies):
    indices = np.arange(len(classwise_accuracies))
    accuracies = dict(sorted(zip(classwise_accuracies, indices)))
    
    plt.xticks(indices)
    plt.ylabel("Accuracy", size='x-large')
    plt.xlabel("Classes", size='x-large')
    plt.bar(indices, list(classwise_accuracies))
    plt.show()