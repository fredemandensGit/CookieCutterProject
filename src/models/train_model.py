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
import datetime

# Setup logging
import logging
date_log = 'logging outputs/'+str(datetime.datetime.now().date())+'/'
logfp = date_log + datetime.datetime.now().strftime('%H: %M') +'/'
result = re.search('(.*).py', os.path.basename(__file__))
result = result.group(1)
os.makedirs(logfp, exist_ok=True)
logging.basicConfig(filename=logfp+result, encoding='utf-8', level=logging.INFO)

# wandb
import wandb




def build_model():
    initialize(config_path="config", job_name="model")
    cfg = compose(config_name="model.yaml")
    logging.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
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
    logging.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    hparams = cfg.hyperparameters

    # set seed
    torch.manual_seed(hparams["seed"])

    logging.info("Training day and night")
    
    
    # initialize wand
    wandb.init(config=cfg)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = torch.load(hparams["data_path"] + "/train_processed.pt")
    train_set = torch.utils.data.DataLoader(
        train_set, batch_size=hparams["batch_size"], shuffle=True
    )
    
    test_set = torch.load(hparams["data_path"] + "/test_processed.pt")
    test_set = torch.utils.data.DataLoader(
        test_set, batch_size=hparams['batch_size_valid'], shuffle=True
    )
    # Setup model, and watch with wandb
    wandb.watch(model, log_freq=100)

    
    criterion = nn.NLLLoss()
    if hparams['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.paramters(), lr=hparams["lr"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    epochs = hparams["epochs"]

    val_images, val_labels = next(iter(test_set))

    train_losses, test_losses = [], []
    model.train()
    for e in range(epochs):
        running_loss = 0

        for images, labels in train_set:
            # View images as vectors
            # images = images.view(images.shape[0], -1)

            # zero gradients
            optimizer.zero_grad()
            
            if sum(model.cl1.weights.grad()) != 0:
                raise ValueError("Weights not set to zero in training loop!")
            
            # Compute loss, step and save running loss
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            # pdb.set_trace()

            running_loss += loss.item()
            
            if running_loss/len(train_set) > 0.4:
                wandb.log({'inputs': wandb.Image(images[0])}) # Wand to extract training images when loss is greater than 0.4 (has no use tbh)

        logging.info(
            "Epoch: {}/{}.. ".format(e + 1, epochs)+
            "Training Loss: {:.3f}.. ".format(running_loss / len(train_set))
        )
            
        wandb.log({"training_loss": running_loss/len(train_set)})
        train_losses.append(running_loss / len(train_set))
    
    log_ps_valid = torch.exp(model(val_images))
    top_p, top_class = log_ps_valid.topk(1, dim=1)
    equals = top_class == val_labels.view(*top_class.shape)
    wandb.log({'validation_accuracy': sum(equals) / len(equals)})

    os.makedirs("reports/figures/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    torch.save(model.state_dict(), "models/trained_model.pt")
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss plot")
    plt.savefig("reports/figures/Training_loss_evolution_config.png", dpi=300)
    torch.save(model, "models/ConvolutionModel_config.pth")


if __name__ == "__main__":
    train()
