"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        super(ConvNet, self).__init__()

        self.conv0 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=1)

        self.preact1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, 3, stride=1, padding=1)
        )
        self.conv1 = nn.Conv2d(64, 128, 1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preact2a = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, stride=1, padding=1)
        )
        self.preact2b = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, stride=1, padding=1)
        )
        self.conv2 = nn.Conv2d(128, 256, 1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preact3a = nn.Sequential(
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )
        self.preact3b = nn.Sequential(
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 256, 3, stride=1, padding=1)
        )
        self.conv3 = nn.Conv2d(256, 512, 1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preact4a = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, stride=1, padding=1)
        )
        self.preact4b = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, stride=1, padding=1)
        )
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.preact5a = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, stride=1, padding=1)
        )
        self.preact5b = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, stride=1, padding=1)
        )
        self.maxpool5 = nn.MaxPool2d(3, stride=2, padding=1)

        self.linear = nn.Sequential(
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Linear(512, 10)
        )

        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        x = self.conv0(x)

        x = self.preact1(x) + x
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.preact2a(x) + x
        x = self.preact2b(x) + x
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.preact3a(x) + x
        x = self.preact3b(x) + x
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.preact4a(x) + x
        x = self.preact4b(x) + x
        x = self.maxpool4(x)

        x = self.preact5a(x) + x
        x = self.preact5b(x) + x
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)
        out = self.linear(x)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
