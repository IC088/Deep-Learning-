'''
Ivan Christian 

Deep Learning HW3 Task 2

FASHIONMNIST Neural Network

'''


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torchvision


# 1 . TODO: Define dataset class and dataloader class


from utils.data_loaders import get_data_loaders


# train_data_loader, test_data_loader = get_data_loaders(100)

# 2 . TODO: Define Model

from models.model import NN
from models.model import Net


# 3 . TODO: Define Loss


# 7 . Train phase function

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.NLLLoss(output, target)
        loss.backward()
        optimizer.step()



# 8 . Validation phase function (using test set as requested)

def evaluate(model, ):
    model.eval()


if __name__ == '__main__':
    print('Still Under Construction')
    # main()