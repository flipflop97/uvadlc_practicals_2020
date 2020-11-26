# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import math


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        # embedding_size = vocabulary_size
        embedding_size = math.ceil(math.log2(vocabulary_size))
        # self.vocabulary_size = vocabulary_size

        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_num_hidden, lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)


    def forward(self, batch_inputs, hidden=None):
        x = self.embedding(batch_inputs)         # (b×t)   -> (b×t×e)
        h, hidden = self.lstm.forward(x, hidden) # (b×t×e) -> (b×t×h)
        p = self.linear(h)                       # (b×t×e) -> (b×t×v)

        p = p.transpose(1, 2) # (b×t×v) -> (b×v×t)

        return p, hidden