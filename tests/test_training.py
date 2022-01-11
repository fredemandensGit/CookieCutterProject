## -*- coding: utf-8 -*-
######################################
############## Imports ###############
######################################
# Data manipulation
import torch
import os

# Graphics
import seaborn as sns

sns.set_style("whitegrid")

import numpy as np

# debugging
import pdb
import pytest

# load model
from src.models.model import MyAwesomeModel

# Import path to data
from tests import _PATH_DATA

# Load model
model = MyAwesomeModel()
model.load_state_dict(torch.load("models/trained_model.pt"))

# pdb.set_trace()

# testing training routine
# def test_has_zero_gradient():
#    with pytest.raises(ValueError, match="Weights not set to zero in training loop!"):
# run something to capture this in training loop - no idea


# Test loading data
@pytest.mark.skipif(
    not (
        os.path.exists("data/processed/train_dataset.pt")
        or os.path.exists("data/processed/test_dataset.pt")
    ),
    reason="Data files not found",
)
def test_load_data():
    torch.load(f"{_PATH_DATA}/processed/test_processed.pt")
