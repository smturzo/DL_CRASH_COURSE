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

class  MnistModel(nn.Module): # Let's dicuss this, not exactly sure whats going on:
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(input_size, num_classes) # here

	def forward(self, xb): # how is the forward function automatically invoked?
		xb = xb.reshape(-1, 784)
		out = self.linear(xb) # and here
		return out

model = MnistModel()
print(model.linear)

for images, labels in train_loader:
	outputs = model(images)
	break

print('outputs.shape : ', outputs.shape) # 128 is batch size and 10 is the ten output as probabilities
print('Sample outputs :\n ', outputs[:2].data)

# SOFTMAX to convert output rows into probabilities
import torch.nn.functional as F
probs = F.softmax(outputs,dim=1)
# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())

# What is inside probs
#print("Probs contain:\n",probs.shape)
max_probs, preds = torch.max(probs, dim=1) #confused about this
print(preds)
print(max_probs)
print(labels)
# Cross entropy loss function
loss_fn = F.cross_entropy
# Loss for current batch of data
loss = loss_fn(outputs, labels)
print(loss)
""" Generic Pseudo code
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        # Generate predictions
        # Calculate loss
        # Compute gradients
        # Update weights
        # Reset gradients
    
    # Validation phase
    for batch in val_loader:
        # Generate predictions
        # Calculate loss
        # Calculate metrics (accuracy etc.)
    # Calculate average validation loss & metrics
    
    # Log epoch, loss & metrics for inspection
"""
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD): #Try Adam Optim # epochs, lr, model, val_loader, train_loader are hyper params
	optimizer = opt_func(model.parameters(), lr)
	history = [] # for recording epoch-wise results
	for epoch in range(epochs):
		# Training Phase 
		for batch in train_loader:
			loss = model.training_step(batch) # training_step not defined yet
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		# Validation phase
		result = evaluate(model, val_loader)
		model.epoch_end(epoch, result)
		history.append(result)

	return history

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader] #validation_step not defined yet.
    return model.validation_epoch_end(outputs)
