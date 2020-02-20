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
import numpy as np


# 1 . TODO: Define dataset class and dataloader class


from utils.data_loaders import get_data_loaders
from utils.vis import plot_training_validation_loss


# train_data_loader, test_data_loader = get_data_loaders(100)

# 2 . TODO: Define Model

from models.model import NN # with cpu
from models.model import Net # with gpu


# 7 . Train phase function

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):

        if isinstance(model, Net):
            data, target = data.to(device), target.to(device)
        else:
            data, target = torch.reshape(data,(data.shape[0],784)).to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # 3. Define the loss (negative log likelihood)
        if isinstance(model, Net):
            loss = F.nll_loss(output, target)
        else:
            loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item()) 
        # print(f'Training Epoch = {epoch} [ {batch_idx * len(data)/len(train_loader.dataset)} ({ batch_idx / len(train_loader):.0%} )')
        # print(f'Loss {loss.item():.6f}')
    if isinstance(model, Net):
        train_loss = torch.mean(torch.tensor(train_losses))
    else:
        train_loss = torch.sum(torch.tensor(np.array(train_losses)))/len(train_loader.dataset)
    print(f'Training Set Average Loss: {train_loss:.4f}')

    return train_loss



# 8 . Validation phase function (using test set as requested)

def evaluate(model, device, val_loader):
    model.eval()

    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            if isinstance(model, Net):
                data, target = data.to(device), target.to(device)
            else:
                data, target = torch.reshape(data,(data.shape[0],784)).to(device), target.to(device)
            output = model(data)
            
            # 3. Define the loss (negative log likelihood)

            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)

    print(f'Validation set Average loss : {val_loss:.4f}')
    print(f'Accuracy : {correct/len(val_loader.dataset)}')

    return val_loss



def main():
    print('Starting Training')
    # 5 . initialize model parameters, usually when model gets instantiated
    batch_size = 64
    epochs = 5
    learningrate = 0.01
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )



    model = Net().to(device)


    
    # 4 . Define the optimiser
    
    optimizer=torch.optim.SGD(model.parameters(),lr=learningrate, momentum=0.0, weight_decay=0)

    train_data_loader, test_data_loader = get_data_loaders(batch_size)


    train_losses, val_losses = [], [] 

    for epoch in range(epochs):
        print (f'current epoch:{epoch+1}')
        print(f'batch size: {batch_size}')

        training_loss = train(model, device, train_data_loader, optimizer, epoch)
        val_loss = evaluate(model, device, test_data_loader) 
        

        # Save a model if val loss is lower than min_loss
        if (len(val_losses) > 0) and (val_loss < min(val_losses)):
            torch.save(model.state_dict(), "fashion_mnist_task2.pt")
            print('Saving Model for testing')

        # Save a list of train and val loss


        train_losses.append(training_loss)
        val_losses.append(val_loss)


    print('Displaying train val graph')
    plot_training_validation_loss(epochs, train_losses, val_losses)


if __name__ == '__main__':
    print('Starting HW3')
    main()