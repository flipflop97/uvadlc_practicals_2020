#!/usr/bin/env python3

import argparse
import numpy as np
from matplotlib import pyplot as plt

from train import train


def main():
    train_steps = 1500
    runs = 3

    for model_type in ['LSTM', 'peepLSTM']:
        for input_length in [10, 20]:
            accuracies = []
            losses = []

            # Train
            for run in range(1, runs + 1):
                print(f"\n{model_type} of length {input_length} - run {run}")

                config = argparse.ArgumentParser()

                config.device = 'cpu'
                config.dataset = 'bipalindrome'
                config.learning_rate = 0.0001
                config.batch_size = 256
                config.num_hidden = 256
                config.max_norm = 10
                config.train_steps = train_steps
                config.model_type = model_type
                config.input_length = input_length

                accuracy, loss = train(config)

                accuracies.append(accuracy)
                losses.append(loss)

            # Calculate
            accuracy_mean = np.mean(accuracies, 0)
            accuracy_std = np.std(accuracies, 0, ddof=1)

            loss_mean = np.mean(losses, 0)
            loss_std = np.std(losses, 0, ddof=1)

            steps = np.arange(accuracy_mean.size)

            # Plot
            plt.figure(figsize=[7.2, 2.4])

            plt.subplot(1, 2, 1)
            plt.xlabel("Step")
            plt.ylabel("Accuracy")
            plt.axhline(1, c="black", alpha=0.3, linewidth=1, zorder=0)
            plt.fill_between(steps, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha=0.2, zorder=1)
            plt.plot(steps, accuracy_mean, zorder=2)
            plt.xlim(steps[0], steps[-1])

            plt.subplot(1, 2, 2)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.axhline(0, c="black", alpha=0.3, linewidth=1, zorder=0)
            plt.fill_between(steps, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, zorder=1)
            plt.plot(steps, loss_mean, zorder=2)
            plt.xlim(steps[0], steps[-1])

            plt.tight_layout()
            plt.savefig(f"plot_{model_type}_{input_length}.png", dpi=400)


if __name__ == "__main__":
    main()