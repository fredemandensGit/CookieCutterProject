import argparse
import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from src.models.model import MyAwesomeModel


class TrainOREvaluate(object):

    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.train()
    train_set = torch.load("data/processed/train_processed.pt")
    train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30
    steps = 0

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0

        for images, labels in train_set:
            # View images as vectors
            # images = images.view(images.shape[0], -1)

            # zero gradients
            optimizer.zero_grad()

            # Compute loss, step and save running loss
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            # pdb.set_trace()

            running_loss += loss.item()

        print(
            "Epoch: {}/{}.. ".format(e + 1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss / len(train_set)),
        )
        train_losses.append(running_loss / len(train_set))

    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss plot")
    plt.savefig("reports/figures/Training_loss_evolution.png", dpi=300)
    torch.save(model, "models/ConvolutionModel_v1_lr0.003_e30_bs64.pth")


if __name__ == "__main__":
    TrainOREvaluate()
