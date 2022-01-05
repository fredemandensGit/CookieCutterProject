import argparse
import logging
import os
import pdb
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim


@hydra.main(config_path="config", config_name="train_conf.yaml")
def main(cfg):
    log = logging.getLogger(__name__)

    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    print(cfg)
    hparams = cfg.hyperparameters

    # set seed
    torch.manual_seed(hparams["seed"])

    log.info("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=hparams["lr"])
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    log.info(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = torch.load(hparams["data_path"] + "/train_processed.pt")
    train_set = torch.utils.data.DataLoader(
        train_set, batch_size=hparams["batch_size"], shuffle=True
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["lr_optim"])

    epochs = hparams["epochs"]

    train_losses, test_losses = [], []
    model.train()
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

        log.info(
            "Epoch: {}/{}.. ".format(e + 1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss / len(train_set)),
        )
        train_losses.append(running_loss / len(train_set))

    os.makedirs("reports/figures/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    torch.save(model.state_dict(), "models/trained_model.pt")
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss plot")
    plt.savefig("reports/figures/Training_loss_evolution.png", dpi=300)
    torch.save(model, "models/ConvolutionModel_v1_lr0.003_e30_bs64.pth")


if __name__ == "__main__":
    main()
