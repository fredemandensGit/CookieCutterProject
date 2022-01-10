## -*- coding: utf-8 -*-
######################################
############## Imports ###############
######################################
# Data manipulation
import torch

# Graphics
import seaborn as sns
sns.set_style("whitegrid")

import numpy as np

# debugging
import pdb

# load model
from src.models.model import MyAwesomeModel

# Load data
Train = torch.load("data/processed/train_processed.pt")
Test = torch.load("data/processed/test_processed.pt")

# Load model
model = MyAwesomeModel()
model.load_state_dict(torch.load("models/trained_model.pt"))

#pdb.set_trace()

# testing model
def test_input_to_output_dims():
    assert np.shape(model(Train[:][0]))[1] == 10, "Shape of model output does not match desired output dimensions"
