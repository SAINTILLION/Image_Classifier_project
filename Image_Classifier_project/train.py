# Imports here
import matplotlib.pyplot as plt
from torchvision import  models
from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np
import os
import argparse
from data_folder import data
import torch.optim as optim
from workspace_utils import active_session



parser = argparse.ArgumentParser()
parser.add_argument("Data_Folder", help="Directory for training dataset", default="flowers/")
parser.add_argument("--arch", default="vgg16", help="model architecture to use")
parser.add_argument("--learning_rate", type=float, default=0.005, help="models learning rate")
parser.add_argument("--hidden_units", help="model hidden units", nargs=2, type=int, default=[4096, 4096])
parser.add_argument("--epochs", help="number of trainig epochs", type=int, default=1)
parser.add_argument("--gpu", help="use gpu for training", default="yes")



args = parser.parse_args()


train_dataLoader, valid_dataLoader, test_dataLoader, train_datasets = data(args.Data_Folder)
First_H, Second_H = args.hidden_units

# Building the model
def create_model(model):
    # freeze the features for the vgg model
    # ie ensures that we do not update the weights of the vgg model when training
    # our own model or rather when performing the back propagation step
    for param in model.parameters():
        param.requires_grad = False

    # define our model classsifier
    classifier = nn.Sequential(nn.Linear(25088, First_H),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.Linear(First_H, Second_H),
                              nn.ReLU(),
                              nn.Dropout(p=0.5),
                              nn.Linear(Second_H, 102),
                              nn.LogSoftmax(dim=1))
                           
    # modify vgg model classifeir
    model.classifier = classifier
    return model

# selects GPU if available for model training
device = torch.device("cuda" if (args.gpu == "yes") else "cpu")

# First import an already trained model
if args.arch == "vgg16":
    model = models.vgg16(pretrained=True)
elif args.arch == "vgg19":
    model = models.vgg19(pretrained=True)

model_1 = create_model(model)


def train_model(model):
    # Trainig process: ie we are training the model with our own classifier
    criterion = nn.NLLLoss()
    # notice that we access only the classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    # moves model to GPU to ensure faster training
    model.to(device)

    epochs = args.epochs
    train_losses, test_losses = [], []
    for e in range(epochs):
        current_loss = 0
        for images, labels in train_dataLoader:


            images = images.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
        else:
            total_test_loss = 0  # Number of incorrect predictions on the test set
            total_test_correct = 0  # Number of correct predictions on the test set


            # turn off gradients
            with torch.no_grad():
                model.eval() # turns of dropouts
                # validation pass here
                for images, labels in valid_dataLoader:


                    images = images.to(device)
                    labels = labels.to(device)


                    output = model(images)
                    loss = criterion(output, labels)
                    total_test_loss += loss.item()

                    predicted_prob = torch.exp(output)
                    top_p, top_class = predicted_prob.topk(1, dim=1)
                    eq = top_class == labels.view(*top_class.shape)
                    total_test_correct += eq.sum().item()

            model.train() # turns dropouts back on
            # Get mean loss to enable comparison between train and test sets
            train_loss = current_loss / len(train_dataLoader.dataset)
            test_loss = total_test_loss / len(valid_dataLoader.dataset)

            # At completion of epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "validation Loss: {:.3f}.. ".format(test_loss),
                  "Model Accuracy: {:.3f}".format(total_test_correct / len(valid_dataLoader.dataset)))
    return model



def save_checkpoint(model):
    model.class_to_idx = train_datasets.class_to_idx
    Checkpoint = {
        "inputs": 25088,
        "outputs": 102,
        "First_hidden_output": First_H,
        "Second_hidden_output": Second_H,
        "class_to_idx": model.class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(Checkpoint, "checkpoint_1.pth")
    print("Model Checkpoint Saved")

    
# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    # Rebuild model
    new_model = models.vgg16(pretrained=True)
    for param in new_model.parameters():
        param.requires_grad = False
    
    new_model.class_to_idx = checkpoint["class_to_idx"]
    
    classifier = nn.Sequential(nn.Linear(checkpoint["inputs"], checkpoint["First_hidden_output"]),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                               nn.Linear(checkpoint["First_hidden_output"], checkpoint["Second_hidden_output"]),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(checkpoint["Second_hidden_output"], checkpoint["outputs"]),
                               nn.LogSoftmax(dim=1))
    
    # update classifier for pretrained network
    new_model.classifier = classifier
    
    new_model.load_state_dict(checkpoint["state_dict"])
    
    return new_model

with active_session():
    #trains our model
    tr_model = train_model(model_1)
    
#save_checkpoint(tr_model)
model_1 = load_checkpoint("checkpoint_1.pth")



