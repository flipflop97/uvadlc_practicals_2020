"""
This module implements a LSTM with peephole connections in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class peepLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(peepLSTM, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.Wfx = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, input_dim, device=device), nonlinearity='sigmoid'))
        self.Wfh = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, hidden_dim, device=device), nonlinearity='sigmoid'))
        self.bf = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.Wix = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, input_dim, device=device), nonlinearity='sigmoid'))
        self.Wih = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, hidden_dim, device=device), nonlinearity='sigmoid'))
        self.bi = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.Wox = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, input_dim, device=device), nonlinearity='sigmoid'))
        self.Woh = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, hidden_dim, device=device), nonlinearity='sigmoid'))
        self.bo = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.Wcx = nn.Parameter(nn.init.kaiming_normal_(torch.empty(hidden_dim, input_dim, device=device), nonlinearity='sigmoid'))
        self.bc = nn.Parameter(torch.zeros(hidden_dim, device=device))

        self.Wph = nn.Parameter(nn.init.kaiming_normal_(torch.empty(num_classes, hidden_dim, device=device), nonlinearity='sigmoid'))
        self.bp = nn.Parameter(torch.zeros(num_classes, device=device))

        self.embedding = nn.Embedding(3, input_dim)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        x = x.long()

        c = torch.zeros(self.batch_size, self.hidden_dim, device=self.device)

        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        softmax = nn.LogSoftmax(1)

        for t in range(self.seq_length):
            xi = self.embedding(x[:, t].squeeze())

            f = sigmoid(xi @ self.Wfx.T + c @ self.Wfh.T + self.bf)
            i = sigmoid(xi @ self.Wix.T + c @ self.Wih.T + self.bi)
            o = sigmoid(xi @ self.Wox.T + c @ self.Woh.T + self.bo)

            c = sigmoid(xi @ self.Wcx.T + self.bc) * i + c * f
            h = tanh(c) * o

        p = h @ self.Wph.T + self.bp
        y = softmax(p)

        return y

        ########################
        # END OF YOUR CODE    #
        #######################

