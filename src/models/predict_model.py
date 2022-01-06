import argparse
import sys

import torch
from torch import nn


class Predict(object):

    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Training arguments")
    # parser.add_argument("load_model_from", default="")
    # add any additional argument that you want
    # args = parser.parse_args(sys.argv[1:])
    # print(args)

    test_set = torch.load("data/processed/test_processed.pt")
    test_set = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model = torch.load("models/ConvolutionModel_v1_lr0.003_e30_bs64.pth")

    criterion = nn.NLLLoss()

    accuracy = []
    with torch.no_grad():
        for images, labels in test_set:

            model.eval()
            # View images as vectors
            # images = images.view(images.shape[0], -1)

            # Evaluation mode - compute loss
            loss_valid = criterion(model(images), labels)

            log_ps_valid = torch.exp(model(images))
            top_p, top_class = log_ps_valid.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy.append(sum(equals) / len(equals))
            
            print(f"Validation loss: {loss_valid.item()}")

    print(f"Mean accuracy: {torch.mean(sum(accuracy)/len(accuracy))*100}%")


if __name__ == "__main__":
    Predict()
