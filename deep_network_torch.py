import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

def download_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data

def createDataloaders(training_data, test_data, BATCH_SIZE = 64 ):
    batch_size = BATCH_SIZE

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    return train_dataloader, test_dataloader

#def main():
print("The value of __name__ is:", repr(__name__)) # this lets us know what is the name of the context of the file, module or class.

if __name__ == "__main__":
    print("executing deep_network_torch.py")
    training_data, test_data = download_data()
    train_dataloader, test_dataloader = createDataloaders(training_data, test_data, BATCH_SIZE = 64 )

