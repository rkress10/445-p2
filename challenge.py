"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Challenge(nn.Module):
    def __init__(self):
        super().__init__()

        ## TODO: define each layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=8,kernel_size=3,stride=2,padding=1)
        self.fc_1 = nn.Linear(in_features=8*1*1,out_features=2)
        self.dropout = nn.Dropout(.001)
        ##

        self.init_weights()

    def init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1/sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0)
        ##

    def forward(self, x):
        N, C, H, W = x.shape

        x = F.relu(self.conv1(x))
        
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(-1,8*1*1)

        #x = self.dropout(x)
        x = self.fc_1(x)

        return x
