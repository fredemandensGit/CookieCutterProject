## -*- coding: utf-8 -*-
######################################
############## Imports ###############
######################################
# Data manipulation
import torch
import os
import pytest

# Graphics
import seaborn as sns
sns.set_style("whitegrid")

import numpy as np

# debugging
import pdb

# Load data
from tests import _PATH_DATA
Train = torch.load("data/processed/train_processed.pt")
Test = torch.load("data/processed/test_processed.pt")

# Number of datapoints in train and test
N_train = 40000
N_test = 5000

# testing
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
def test_N_observations():
    Train = torch.load("data/processed/train_processed.pt")
    Test = torch.load("data/processed/test_processed.pt")
    assert len(Train) == N_train, "Train dataset does not contain expected number of observations"
    assert len(Test) == N_test, "Test dataset does not contain expected number of observations"

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
def test_Int_Labels():
    Train = torch.load("data/processed/train_processed.pt")
    Test = torch.load("data/processed/test_processed.pt")
    for i in range(len(Train)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor, "Train data not a tensor"
        assert type(Train.__getitem__(i)[1].item()) == int, "Train labels not integer"

    for i in range(len(Test)):
        assert type(Train.__getitem__(i)[1]) == torch.Tensor, "Test data not a tensor"
        assert type(Test.__getitem__(i)[1].item()) == int, "Test labels not integer"

@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
def test_Classes_Represented():
    Train = torch.load("data/processed/train_processed.pt")
    Test = torch.load("data/processed/test_processed.pt")
    assert (torch.unique(Train[:][1]).numpy() == np.arange(0,10)).all(), "All classes not represented in train data"
    assert (torch.unique(Test[:][1]).numpy() == np.arange(0,10)).all(), "All classes not represented in test data"
    
@pytest.mark.skipif(not os.path.exists(f'{_PATH_DATA}/processed/test_processed.pt'), reason="Data files not found")
def test_DataShape():
    Train = torch.load("data/processed/train_processed.pt")
    Test = torch.load("data/processed/test_processed.pt")
    for i in range(len(Train)):
        assert Train.__getitem__(i)[0].shape[0] == 28, "Image shape not correct in train data"
        assert Train.__getitem__(i)[0].shape[1] == 28, "Image shape not correct in train data"

    for i in range(len(Test)):
        assert Test.__getitem__(i)[0].shape[0] == 28, "Image shape not correct in test data"
        assert Test.__getitem__(i)[0].shape[1] == 28, "Image shape not correct in train data"

def test_load_data():
    torch.load(f'{_PATH_DATA}/processed/test_processed.pt')
