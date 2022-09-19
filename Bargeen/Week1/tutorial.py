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
train_ds, val_ds = random_split(dataset, [50000, 10000]) #Hyper parameters
print("Size of training set: ", len(train_ds), "Size of validation set: ", len(val_ds))

from torch.utils.data import DataLoader
batch_size = 128 #Hyper parameter
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size) 
# On the other hand, since the validation data loader is used only for evaluating the model, 
# There is no need to shuffle the images. (Not sure I understand this explanation completely)

import torch.nn as nn

input_size = 28*28 # This is to flatten the 28 by 28 image into a input vector of 784 items
num_classes = 10   # Since it has a probability for images from 0 to 9, we need 10 here. Confused about this.

# Logistic regression model
model = nn.Linear(input_size, num_classes)


print("Total number of model weights: ",model.weight.shape)
print("Weights of the model: ")
print(model.weight)
print("Total number of biases")
print(model.bias.shape)
print("Biases of the model:")
print(model.bias)
