"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    accuracy = (torch.argmax(predictions, 1) == torch.argmax(targets, 1)).float().mean()

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    dd = FLAGS.data_dir
    sm = FLAGS.max_steps
    lr = FLAGS.learning_rate
    sb = FLAGS.batch_size
    fe = FLAGS.eval_freq

    cifar10 = cifar10_utils.get_cifar10(dd)

    dnn_inputs = cifar10['test'].images[0].size
    dnn_classes = cifar10['test'].labels[0].size

    st = cifar10['test'].labels.shape[0]

    x_test, t_test = cifar10['test'].images, cifar10['test'].labels
    x_test = x_test.reshape(st, dnn_inputs)
    x_test = torch.tensor(x_test)
    t_test = torch.tensor(t_test)

    dnn = MLP(dnn_inputs, dnn_hidden_units, dnn_classes)
    ce = nn.CrossEntropyLoss()

    optim = torch.optim.SGD(dnn.parameters(), lr=lr)

    losses = []
    loss_steps = []
    accuracies = []
    accuracy_steps = []

    for step in range(sm):
        optim.zero_grad()

        # Preprocessing
        x_train, t_train = cifar10['train'].next_batch(sb)
        x_train = x_train.reshape(sb, dnn_inputs)
        x_train = torch.tensor(x_train)
        t_train = torch.tensor(t_train)

        # Forward pass
        y_train = dnn.forward(x_train)
        loss = ce.forward(y_train, t_train.argmax(1))

        # Calculate gradients
        loss.backward()

        # Evaluate
        losses.append(loss)
        loss_steps.append(step)

        if step % fe == 0 or sm - step == 1:
            with torch.no_grad():
                y_test = dnn.forward(x_test)
                acc = accuracy(y_test, t_test)

                accuracies.append(acc)
                accuracy_steps.append(step)

                print(f"{step/sm*100:3.0f}%\tLoss {loss:.3f}\tAccuracy {acc:.3f}")

        # Update parameters
        optim.step()


    try:
        from matplotlib import pyplot as plt

        plt.figure(figsize=[7.2, 2.4])

        plt.subplot(1, 2, 1)
        plt.plot(loss_steps, losses)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.xlim(loss_steps[0], loss_steps[-1])

        plt.subplot(1, 2, 2)
        plt.plot(accuracy_steps, accuracies)
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.xlim(accuracy_steps[0], accuracy_steps[-1])

        plt.tight_layout()
        plt.savefig("plot_mlp_pytorch.png", dpi=400)

    except ModuleNotFoundError:
        pass    

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
