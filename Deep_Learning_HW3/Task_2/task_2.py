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
import torch.nn.functional as F


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
        # data = data.view(data.size(0), -1)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item()) 
        print(f'Training Epoch = {epoch} [ {batch_idx * len(data)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0%}% )')
        print(f'Loss {loss.item():.6f}')
    train_loss = torch.mean(torch.tensor(train_losses))
    print(f'Training Set Average Loss: {train_loss:.4f}')



# 8 . Validation phase function (using test set as requested)

def evaluate(model, device, val_loader):
    model.eval()

    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # sum up all the batch losses
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)

    print(f'Validation set Average loss : {val_loss:.4f}')
    print(f'Accuracy : {correct}/{len(val_loader.dataset)}')






def main():
    print('Starting Training')

    batch_size = 64
    epochs = 100
    learningrate = 0.01
    device = torch.device( 'cpu' )

    model = NN()
    optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)

    train_data_loader, test_data_loader = get_data_loaders(batch_size)

    for epoch in range(epochs):
        print ('current epoch:', epoch+1)
        training_loss = train(model, device, train_data_loader, optimizer, epoch)
    print(training_loss)



if __name__ == '__main__':
    print('Still Under Construction')
    main()