# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    raw_data_path = "data/raw/"

    # Dictionaries
    test = {"images": [], "labels": []}
    train = {"images": [], "labels": []}

    # Load images and labels
    for i in range(0, 5):
        file = "train_" + str(i) + ".npz"
        with np.load(raw_data_path + file) as f:
            train["images"].append(f["images"])
            train["labels"].append(f["labels"])

    with np.load(raw_data_path + "test.npz") as f:
        test["images"].append(f["images"])
        test["labels"].append(f["labels"])

    # convert to appropiate dimensions
    ims = np.array(train["images"])
    m, n = ims.shape[2:4]
    train["images"] = ims.reshape(-1, m, n)
    train["labels"] = np.array(train["labels"]).flatten()

    ims = np.array(test["images"])
    m, n = ims.shape[2:4]
    test["images"] = ims.reshape(-1, m, n)
    test["labels"] = np.array(test["labels"]).flatten()

    # Convert to data loader by using tensor datasets
    train = torch.utils.data.TensorDataset(
        torch.Tensor(train["images"]), torch.LongTensor(train["labels"])
    )
    # train = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    test = torch.utils.data.TensorDataset(
        torch.Tensor(test["images"]), torch.LongTensor(test["labels"])
    )
    # test = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)

    # Save file
    save_path = "data/processed/"

    torch.save(train, save_path + "train_processed.pt")
    torch.save(test, save_path + "test_processed.pt")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
