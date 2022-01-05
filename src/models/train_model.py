import argparse
import re
import sys

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim

import os
import pdb
from model import MyAwesomeModel, CNNModuleVar
sns.set_style("whitegrid")

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf,DictConfig


import logging
log = logging.getLogger(__name__)

def build_model():
    initialize(config_path="config", job_name="model")
    cfg = compose(config_name="model.yaml")
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    configs = cfg['hyperparameters']

    ###################################################
    ################# Hyperparameters #################
    ###################################################
    input_channel = configs['input_channel']
    conv_to_linear_dim = configs['conv_to_linear_dim']
    output_dim = configs['output_dim']
    hidden_channel_array = configs['hidden_channel_array']
    hidden_kernel_array = configs['hidden_kernel_array']
    hidden_stride_array = configs['hidden_stride_array']
    hidden_padding_array = configs['hidden_padding_array']
    hidden_dim_array = configs['hidden_dim_array']
    non_linear_function_array = configs['non_linear_function_array']
    regularization_array = configs['regularization_array']

    # Define models, loss-function and optimizer
    model = CNNModuleVar(input_channel, conv_to_linear_dim,
                        output_dim,hidden_channel_array,
                        hidden_kernel_array,hidden_stride_array,
                        hidden_padding_array,hidden_dim_array,
                        non_linear_function_array,regularization_array)

    return model

def train():
    # Get model struct
    model = build_model()

    cfg = compose(config_name="train_conf.yaml")
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

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
    plt.savefig("reports/figures/Training_loss_evolution_config.png", dpi=300)
    torch.save(model, "models/ConvolutionModel_v1_lr0.003_e30_bs64_config.pth")


if __name__ == "__main__":
    train()
