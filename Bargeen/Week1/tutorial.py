#imports
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

# Download training dataset
dataset = MNIST(root='data/', download=True)
print("Number of Images in dataset: ",len(dataset))
test_dataset = MNIST(root='data/', train=False)
print("Number of Images in dataset for test set",len(test_dataset))

print("This is the first image of the dataset: ",dataset[0])
import torchvision.transforms as transforms
# MNIST dataset (images and labels)
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())
img_tensor, label = dataset[0]
print(img_tensor.shape, label)

from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [50000, 10000])
print("Size of training set: ", len(train_ds), "Size of validation set: ", len(val_ds))

