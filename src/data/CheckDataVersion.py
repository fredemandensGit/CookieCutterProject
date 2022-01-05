#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:47:09 2022

@author: frederikhartmann
"""
import numpy as np
import torch

## Check processed data
train_set = torch.load("data/processed/train_processed.pt")
# train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torch.load("data/processed/test_processed.pt")
# test_set = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

images_train, labels_train = train_set[:]
images_test, labels_test = test_set[:]

print("Training images data dimensions: ", images_train.shape)
print("Test images data dimensions: ", images_test.shape)
