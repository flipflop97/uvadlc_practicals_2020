#!/usr/bin/env python3

import argparse
import numpy as np
from matplotlib import pyplot as plt

from train import train


def main():
    # Train
    config = argparse.ArgumentParser()

    config.device = 'cpu'
    config.learning_rate = 2e-3
    config.batch_size = 64
    config.lstm_num_hidden = 128
    config.lstm_num_layers = 2
    config.train_steps = int(1e6)
    config.print_every = 1422
    config.sample_every = 7110
    config.seq_length = 30
    config.temp = 2
    config.txt_file = "assets/book_NL_darwin_reis_om_de_wereld.txt"

    accuracy, loss = train(config)

    steps = np.arange(len(accuracy))

    # Plot
    plt.figure(figsize=[7.2, 2.4])

    plt.subplot(1, 2, 1)
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.axhline(1, c="black", alpha=0.3, linewidth=1, zorder=0)
    plt.plot(steps, accuracy, zorder=1)
    plt.xlim(steps[0], steps[-1])

    plt.subplot(1, 2, 2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.axhline(0, c="black", alpha=0.3, linewidth=1, zorder=0)
    plt.plot(steps, loss, zorder=1)
    plt.xlim(steps[0], steps[-1])

    plt.tight_layout()
    plt.savefig(f"plot_generative.png", dpi=400)


if __name__ == "__main__":
    main()