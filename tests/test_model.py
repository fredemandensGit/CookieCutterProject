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

# Testing 
import pytest

# Load data
Train = torch.load("data/processed/train_processed.pt")
Test = torch.load("data/processed/test_processed.pt")

# Load model
model = MyAwesomeModel()
model.load_state_dict(torch.load("models/trained_model.pt"))

#pdb.set_trace()

# testing model
@pytest.mark.parametrize("batch", [20, 30, 50])
def test_input_to_output_dims(batch): # pragma: no cover
    # Load data and set batch size shuffle
    Train = torch.load("data/processed/train_processed.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=batch, shuffle=True)
    
    # take images, labels
    images, labels = next(iter(train_set))
    
    # run through model
    log_ps = model(images)
    
    # asswer sizes
    assert log_ps.shape[0] == batch, "Shape of model output does not match batch size"
    assert log_ps.shape[1] == 10, "Shape of model output does not match number of labels"

def test_data_shape():
    with pytest.raises(ValueError, match="Expected input to be a 3D tensor"):
        model(torch.randn(1, 64, 28, 28))
    
def test_image_dim():
    with pytest.raises(ValueError, match=r"Expected each sample to have shape \[28, 28\]"):
        model(torch.randn(1,2,3))
    
from tests import _PATH_DATA

def test_load_data():
    torch.load(f'{_PATH_DATA}/processed/test_processed.pt')
