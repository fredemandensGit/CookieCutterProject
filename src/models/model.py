import logging

import hydra
from omegaconf import OmegaConf
from torch import nn


class MyAwesomeModel(nn.Module):
    @hydra.main(config_path="config", config_name="ConvolutionModel_v1_lr0.003_e30_bs64_conf.yaml")
    def main(cfg):
        log = logging.getLogger(__name__)

        log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

        hparams = cfg.hyperparameters

        # set seed

        def __init__(self):
            super().__init__()

            # Convolutional layers -
            self.conv1 = nn.Conv2d(
                in_channels=hparams["in_1"],
                out_channels=hparams["out_1"],
                kernel_size=hparams["kernel_size"],
                stride=hparams["stride"],
                padding=hparams["padding"],
            )
            self.conv2 = nn.Conv2d(
                in_channels=hparams["in_2"],
                out_channels=hparams["out_1"],
                kernel_size=hparams["kernel_size"],
                stride=hparams["stride"],
                padding=hparams["padding"],
            )
            self.conv3 = nn.Conv2d(
                in_channels=hparams["in_3"],
                out_channels=hparams["out_1"],
                kernel_size=hparams["kernel_size"],
                stride=hparams["stride"],
                padding=hparams["padding"],
            )

            # Output - one for each digit
            self.fc1 = nn.Linear(
                hparams["out_1"] * hparams["kernel_size"] * hparams["kernel_size"],
                hparams["lin_out_1"],
            )
            self.fc2 = nn.Linear(hparams["lin_out_1"], hparams["lin_out_2"])
            self.out = nn.Linear(hparams["lin_out_2"], hparams["out_digits"])

            # Define sigmoid activation and softmax output
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.log_softmax = nn.LogSoftmax(dim=1)

            # Dropout
            self.dropout = nn.Dropout(p=hparams["dropout_p"])

            # max pool
            self.maxpool = nn.MaxPool2d((2, 2))

        def forward(self, x):
            # Pass the input tensor through each of our operations
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])  # correct view

            # convolutional layers
            x = self.maxpool(self.leaky_relu(self.conv1(x)))
            x = self.maxpool(self.leaky_relu(self.conv2(x)))
            x = self.maxpool(self.leaky_relu(self.conv3(x)))

            # Flatten x to send through fully connected layers
            x = x.view(x.shape[0], -1)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.log_softmax(self.out(x))
            return x
