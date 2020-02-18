import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
'''
1 . TODO: Define dataset class and dataloader class
'''


def get_data_loaders(batch_size):
	train_set = FashionMNIST(root = "./data", train = True, download = True, transform = transforms.ToTensor())
	test_set = FashionMNIST(root = "./data", train = False, download = True, transform = transforms.ToTensor())
	
	train_data_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)
	test_data_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=4)

# datasets.FashionMNIST('../data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
	return train_data_loader, test_data_loader