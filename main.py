import argparse
import pdb
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim

from data import mnist


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        # print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        epochs = 30
        steps = 0

        train_losses, test_losses = [], []
        for e in range(epochs):
            running_loss = 0

            for images, labels in train_set:
                # View images as vectors
                images = images.view(images.shape[0], -1)

                # zero gradients
                optimizer.zero_grad()

                # Compute loss, step and save running loss
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                steps += 1
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
        torch.save(model.state_dict(), "model.pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)

        _, test_set = mnist()
        test_set = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

        criterion = nn.NLLLoss()

        accuracy = []
        with torch.no_grad():
            for images, labels in test_set:
                model.eval()
                # View images as vectors
                images = images.view(images.shape[0], -1)

                # Evaluation mode - compute loss
                loss_valid = criterion(model(images), labels)

                log_ps_valid = torch.exp(model(images))
                top_p, top_class = log_ps_valid.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                accuracy.append(sum(equals) / len(equals))
                print(f"Validation loss: {loss_valid.item()}")

        print(f"Mean accuracy: {torch.mean(sum(accuracy)/len(accuracy))*100}%")


if __name__ == "__main__":
    TrainOREvaluate()
