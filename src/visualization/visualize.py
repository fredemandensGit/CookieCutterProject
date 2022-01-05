import argparse
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Graphics
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import nn
from torchsummary import summary

from src.models.model import MyAwesomeModel

sns.set_style("whitegrid")
sns.set_theme()
# Debuging
import pdb

from matplotlib import cm


class Visualizer(object):

    ###################################################
    #################### Arguments ####################
    ###################################################
    # Arguments to be called
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--load_model_from", default="models/")
    parser.add_argument("--modelName", default="ConvolutionModel_v1_lr0.003_e30_bs64.pth")

    # Save arguments in args
    args = parser.parse_args(sys.argv[1:])
    print(args)

    ###################################################
    ############### Load model and data ###############
    ###################################################
    # Load model
    model = torch.load(args.load_model_from + args.modelName)
    model.eval()

    # Extract model name
    result = re.search("(.*).pth", args.modelName)
    modelName = result.group(1)

    # Folderpath to images
    fpath = "reports/figures/features/"

    # Load data
    Train = torch.load("data/processed/train_processed.pt")
    train_set = torch.utils.data.DataLoader(Train, batch_size=Train.__len__(), shuffle=True)

    ###################################################
    ############# Plot histogram of data ##############
    ###################################################
    _, labels = Train[:]
    plt.hist(labels.numpy(), bins=np.arange(11) - 0.5, edgecolor="red", facecolor="black")
    plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
    plt.xlabel("Number in MNIST image")
    plt.ylabel("Number of occurences")
    plt.title("Label distribution of handwritten MNIST digits")
    plt.savefig("reports/figures/label_distribution.png", dpi=300)

    # Plot the features from the convolutional layer before the first fully connected layer
    # Make sure all number are present in batch (and find index of first number x)
    allNumber = True
    index0to9 = []
    while allNumber:
        allNumber = False
        index0to9 = []

        # Extract batch of data
        (images, labels) = next(iter(train_set))
        for i in range(10):
            try:
                index0to9.append(labels.tolist().index(i))
            except:
                allNumber = True

    # Extract activations from model and save in activations
    def get_activation(name):
        activation = {}

        def hook(model, input, output):
            activation[name] = output.detach()

        return hook, activation

    # Extract all cnn layers
    activation = {}
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            hook, activ = get_activation(name)
            activation[name] = activ
            layer.register_forward_hook(hook)  # use hook
    output = model(images)

    # Make activation dictionary a single dictionary
    active = Counter()
    for i in activation.values():
        active.update(i)

    # Get activations of conv2D layers
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):  # Only plot activations of convolutions
            num_filters = active[name].shape[1]
            num_rows = int(np.floor(num_filters / 4))

            # Create path to images, if not existing
            pathToImage = fpath + modelName + "/" + name + "/"
            isExist = os.path.exists(pathToImage)
            if not isExist:
                os.makedirs(pathToImage)

            for i in range(10):
                # Get activations of i'th number
                activ = active[name][index0to9[i]]
                activ = (activ - torch.min(activ)) / (torch.max(activ) - torch.min(activ))

                # Define grid of plots
                f, axarr = plt.subplots(nrows=num_rows, ncols=4, figsize=(4 * 3, num_rows * 3))
                if num_rows > 1:
                    for j in range(num_filters):
                        axarr[int(np.floor(j / 4))][j % 4].imshow(activ[j])
                        axarr[int(np.floor(j / 4))][j % 4].set_title("Layer number: " + str(j))
                else:
                    for j in range(num_filters):
                        axarr[j].imshow(activ[j])
                        axarr[j].set_title("Layer number: " + str(j))

                # Save images
                plt.savefig(pathToImage + "number" + str(i) + ".png")

    ###################################################
    ############ t-SNE plots final layer ##############
    ###################################################
    # Inspiration from https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
    # Get activations of final layer (embedding)
    layer = list(model.children())[4]
    final_hook, final_active = get_activation("final")
    print(final_hook)
    print(final_active)
    layer.register_forward_hook(final_hook)
    output = model(images)

    # Extract embedding of all images with their respective predictions
    # Initialize arrays
    test_imgs = torch.zeros((0, 28, 28), dtype=torch.float32)
    test_predictions = []
    test_targets = []
    test_embeddings = torch.zeros((0, 64), dtype=torch.float32)

    for x, y in train_set:
        # Get top 1 probs per image
        log_ps = model(x)
        ps = torch.exp(log_ps)
        top_p, preds = ps.topk(1, dim=1)
        # Extract embedding
        embeddings = final_active["final"]
        # Populate arrays
        test_predictions.extend(preds.detach().tolist())
        test_targets.extend(y.detach().tolist())
        test_embeddings = torch.cat((test_embeddings, embeddings.detach()), 0)
        test_imgs = torch.cat((test_imgs, x.detach()), 0)
    # Transform arrays to numpy
    test_imgs = np.array(test_imgs)
    test_embeddings = np.array(test_embeddings)
    test_targets = np.array(test_targets)
    test_predictions = np.array(test_predictions)
    test_predictions = test_predictions.squeeze()

    # Run t-SNE on the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)

    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(
            tsne_proj[indices, 0],
            tsne_proj[indices, 1],
            c=np.array(cmap(lab)).reshape(1, 4),
            label=lab,
            alpha=0.5,
        )
    ax.legend(fontsize="large", markerscale=2)

    # Save image
    plt.savefig(fpath + modelName + "/t-SNE_embedded.png")


if __name__ == "__main__":
    Visualizer()
