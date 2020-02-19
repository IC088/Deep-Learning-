'''
Task 2 Model definition

Making the program less cluttered by creating classes and functions seperately.
'''



import torch.nn as nn
import torch.nn.functional as F

'''
2 .  TODO: Define Model

Defining the model as specified in the HW documentation 
'''
#Not required in the end since using gpu, but good practice to be defined for first time

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(28, 300) # 300 Neurons for first layer, from 28 input from the dataset
        self.fc2 = nn.Linear(300, 100) # 100 Neurons for second layer
        self.fc3 = nn.Linear(100, 10)  # 10 neurons for 3rd layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)
'''
Testing using GPU 
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)