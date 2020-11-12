"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    dd = FLAGS.data_dir
    sm = FLAGS.max_steps
    lr = FLAGS.learning_rate
    sb = FLAGS.batch_size
    fe = FLAGS.eval_freq

    test_batches = 8

    cifar10 = cifar10_utils.get_cifar10(dd)

    dnn_inputs = cifar10['test'].images[0].shape[0]
    dnn_classes = cifar10['test'].labels[0].size

    st = cifar10['test'].labels.shape[0]

    x_test, t_test = cifar10['test'].images, cifar10['test'].labels
    x_test_list = [torch.tensor(x) for x in np.split(x_test, test_batches)]
    t_test_list = [torch.tensor(t) for t in np.split(t_test, test_batches)]

    dnn = ConvNet(dnn_inputs, dnn_classes)
    ce = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(dnn.parameters(), lr=lr)

    losses = []
    loss_steps = []
    accuracies = []
    accuracy_steps = []

    acc_best = 0
    loss_best = 0
    step_best = 0

    for step in range(sm):
        optim.zero_grad()

        # Preprocessing
        x_train, t_train = cifar10['train'].next_batch(sb)
        x_train = torch.tensor(x_train)
        t_train = torch.tensor(t_train)

        # Forward pass
        y_train = dnn.forward(x_train)
        loss = ce.forward(y_train, t_train.argmax(1))

        # Calculate gradients
        loss.backward()

        # Evaluate
        losses.append(loss.detach())
        loss_steps.append(step)

        if step % fe == 0 or sm - step == 1:
            with torch.no_grad():
                acc = 0                
                for i in range(test_batches):
                    y_test_i = dnn.forward(x_test_list[i])
                    acc += accuracy(y_test_i, t_test_list[i]).detach()
                acc /= test_batches

                accuracies.append(acc)
                accuracy_steps.append(step)

                if acc > acc_best:
                    acc_best = acc
                    loss_best = loss
                    step_best = step

                print(f"{step/sm*100:3.0f}% - Loss {loss:.3f} - Accuracy {acc:.3f} - Best {acc_best:.3f}")


        # Update parameters
        optim.step()


    try:
        from matplotlib import pyplot as plt

        plt.figure(figsize=[7.2, 2.4])

        plt.subplot(1, 2, 1)
        plt.plot(loss_steps, losses)
        plt.axvline(step_best, c="black", alpha=0.3, linewidth=1)
        plt.axhline(loss_best, c="black", alpha=0.3, linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.xlim(loss_steps[0], loss_steps[-1])

        plt.subplot(1, 2, 2)
        plt.plot(accuracy_steps, accuracies)
        plt.axvline(step_best, c="black", alpha=0.3, linewidth=1)
        plt.axhline(acc_best, c="black", alpha=0.3, linewidth=1)
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.xlim(accuracy_steps[0], accuracy_steps[-1])

        plt.tight_layout()
        plt.savefig("plot_convnet_pytorch.png", dpi=400)

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
