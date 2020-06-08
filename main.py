import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from LeNet import *
from train_val import *
from torch.utils.data.sampler import SubsetRandomSampler
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 1


# define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# 6 and 8 classes from mnist
train_subset_indices = ((train_dataset.train_labels == 6) + (train_dataset.train_labels == 8)).nonzero().view(-1)

valid_subset_indices = ((valid_dataset.test_labels == 6) + (valid_dataset.test_labels == 8)).nonzero().view(-1)


train_dataset.targets[train_dataset.targets == 6] = 0.0
train_dataset.targets[train_dataset.targets == 8] = 1.0

valid_dataset.targets[valid_dataset.targets == 6] = 0.0
valid_dataset.targets[valid_dataset.targets == 8] = 1.0


# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                        #   shuffle = True
                          sampler=SubsetRandomSampler(train_subset_indices)
                         )

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE,
                        #   shuffle = False
                          sampler=SubsetRandomSampler(valid_subset_indices)
                         )

# # define the data loaders
# train_loader = DataLoader(dataset=train_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True)

# valid_loader = DataLoader(dataset=valid_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=False)


# plot the dataset
ROW_IMG = 10
N_ROWS = 5

fig = plt.figure()
for index in range(1, ROW_IMG * N_ROWS + 1):
    plt.subplot(N_ROWS, ROW_IMG, index)
    plt.axis('off')
    plt.imshow(train_dataset.data[index])
fig.suptitle('MNIST Dataset - preview')
print("aho")

# instantiate the model
torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# start training
model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)


