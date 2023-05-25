from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder and apply the transforms to it
    train_datasets =  datasets.ImageFolder(train_dir, transform=train_transforms) # Loading the training datasets

    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms) # Loading the validation datasets

    test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms) # Loading the test datasets


    # Using the image datasets and the transforms, define the dataloaders
    train_dataLoader = DataLoader(train_datasets, batch_size=64, shuffle=True)

    valid_dataLoader = DataLoader(valid_datasets, batch_size=32)

    test_dataLoader = DataLoader(test_datasets, batch_size=32)

    
    return train_dataLoader, valid_dataLoader, test_dataLoader, train_datasets

