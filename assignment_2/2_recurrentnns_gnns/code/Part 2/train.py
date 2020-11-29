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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################


def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    max_steps = min(config.train_steps, len(data_loader) - 1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        config.batch_size,
        config.seq_length,
        dataset.vocab_size,
        config.lstm_num_hidden,
        config.lstm_num_layers,
        device
    )

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config.learning_rate)

    accuracy_list = []
    loss_list = []

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        batch_inputs = torch.stack(batch_inputs).T   # (t×[b]) -> (b×t)
        batch_targets = torch.stack(batch_targets).T # (t×[b]) -> (b×t)

        batch_predictions, _ = model.forward(batch_inputs)

        loss = criterion(batch_predictions, batch_targets)
        accuracy = (batch_predictions.detach().argmax(1) == batch_targets).float().mean()

        loss.backward()
        optimizer.step()

        accuracy_list.append(accuracy.item())
        loss_list.append(loss.item())

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0 or step == max_steps:
            print(
                "[{}] Train Step {:04d}/{:04d}, "
                "Batch Size = {}, Examples/Sec = {:.2f}, "
                "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    step, max_steps, config.batch_size,
                    examples_per_second, accuracy, loss
                )
            )

        if step % config.sample_every == 0 or step == max_steps:
            with torch.no_grad():
                # Random first character
                sample_hidden = None
                sample_inputs = torch.randint(dataset.vocab_size, (5, 1))
                sentences = sample_inputs.clone()

                for i in range(69):
                    sample_predictions, sample_hidden = model.forward(sample_inputs, sample_hidden)
                    if config.temp:
                        sample_probabilities = torch.softmax(config.temp * sample_predictions, 1)[:,:,0]
                        sample_inputs = torch.multinomial(sample_probabilities, 1, True)
                    else:
                        sample_inputs = sample_predictions.argmax(1)
                    sentences = torch.hstack((sentences, sample_inputs))

                for sentence in sentences:
                    print(dataset.convert_to_string(sentence.tolist()).replace("\n", "|"))

                # Sentences from book
                sample_hidden = None
                sample_inputs = torch.tensor([
                    [dataset._char_to_ix[c] for c in "Tijdens mijn verblijf op deze "],
                    [dataset._char_to_ix[c] for c in "Ik vond niet minder dan 12 ver"],
                    [dataset._char_to_ix[c] for c in "Het eerst bezocht ik het woud,"],
                    [dataset._char_to_ix[c] for c in "Op mijne wandelingen stak ik m"],
                    [dataset._char_to_ix[c] for c in "Tot besluit schijnt het mij to"]
                ])
                sentences = sample_inputs.clone()

                for i in range(40):
                    sample_predictions, sample_hidden = model.forward(sample_inputs, sample_hidden)
                    if config.temp:
                        sample_probabilities = torch.softmax(config.temp * sample_predictions, 1)[:,:,0]
                        sample_inputs = torch.multinomial(sample_probabilities, 1, True)
                    else:
                        sample_inputs = sample_predictions.argmax(1)
                    sentences = torch.hstack((sentences, sample_inputs))

                for sentence in sentences:
                    print(dataset.convert_to_string(sentence.tolist()).replace("\n", "|"))


        if step == config.train_steps:
            break

    print('Done training.')

    return accuracy_list, loss_list

###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments
    parser.add_argument('--temp', type=float, default=None, help='Temperature of random sampling')

    config = parser.parse_args()

    # Train the model
    train(config)
