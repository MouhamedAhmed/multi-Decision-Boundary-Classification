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
from contrastive_loss import *
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10
contrastive_ratio = 1.0
margin = 32


# instantiate the model
torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cross_entropy_loss_criterion = nn.CrossEntropyLoss()
contrastive_loss_criterion = ContrastiveLoss()

# start training
model, optimizer, train_losses, valid_losses = training_loop(model, cross_entropy_loss_criterion,contrastive_loss_criterion,BATCH_SIZE, optimizer, N_EPOCHS,contrastive_ratio,margin, DEVICE)
print(train_losses)
print(valid_losses)

