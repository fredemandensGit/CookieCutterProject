import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn

from src.models.model import MyAwesomeModel


class Visualizer(object):

    model = MyAwesomeModel()
    train_set = torch.load("data/processed/train_processed.pt")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    state_dict = torch.load("models/model.pt")
    model.load_state_dict(state_dict)

    # Data distribution - histogram of labels
    _, labels = train_set[:]
    plt.hist(
        labels.numpy(), bins=np.arange(11) - 0.5, edgecolor="red", facecolor="black"
    )
    plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
    plt.xlabel("Number in MNIST image")
    plt.ylabel("Number of occurences")
    plt.title("Label distribution of handwritten MNIST digits")
    plt.savefig("reports/figures/label_distribution.png", dpi=300)

    # Plot the features from the convolutional layer before the first fully connected layer
    print(model.conv1)


if __name__ == "__main__":
    Visualizer()
