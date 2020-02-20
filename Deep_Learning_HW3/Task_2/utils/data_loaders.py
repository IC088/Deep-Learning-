import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
'''
1 . TODO: Define dataset class and dataloader class
'''


def get_data_loaders(batch_size):
    train_set = FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
    test_set = FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())
    
    
    test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=4)


    val_set = FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())

    # Getting the validation set from the training set
    valid_size = 0.05
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))


    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)



    train_data_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,sampler=train_sampler,num_workers=4)
    val_data_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,sampler=val_sampler,num_workers=4)

    return train_data_loader, test_data_loader, val_data_loader


def test_accuracy(confusion_matrix):
    classwise_accuracies = confusion_matrix.diag()/confusion_matrix.sum(1)
    return classwise_accuracies