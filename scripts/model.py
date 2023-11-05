import matplotlib.pyplot as plt # for plotting
import numpy as np # for transformation

import torch # PyTorch package
import torchvision # load datasets
import torchvision.transforms as transforms # transform data
import torch.nn as nn # basic building block for neural neteorks
import torch.nn.functional as F # import convolution functions like Relu
import torch.optim as optim # optimzer


class CNN_Net(nn.Module):
    # ''' Models a simple Convolutional Neural Network'''
	
    def __init__(self):
	    # ''' initialize the network '''
        super(CNN_Net, self).__init__()
        # 3 input image channel, 6 output channels, 
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
	    # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 1, 5) 
        self.fc1 = nn.Linear(18369, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.mlayer1 = nn.Linear(16,64)
        self.mlayer2 = nn.Linear(64, 16)
        self.mlayer3 = nn.Linear(16, 1)


        

    def forward(self, x,y):
	    # ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        y = self.mlayer1(y)
        y = self.mlayer2(y)
        y = self.mlayer3(y)
        # print(x)
        # print(y)
        return abs(x + y)



