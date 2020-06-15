import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from LeNet import *
from ResNet50 import *
from train_val import *
from torch.utils.data.sampler import SubsetRandomSampler
from contrastive_loss import *
from cosine_contrastive_loss import *
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
N_EPOCHS = 30

IMG_SIZE = 32
N_CLASSES = 200
margin = 2

print('contrastive ratio:')
contrastive_ratio = float(input())
# contrastive_ratio = 0.5

# instantiate the model
torch.manual_seed(RANDOM_SEED)

model = ResNet50(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cross_entropy_loss_criterion = nn.CrossEntropyLoss()
contrastive_loss_criterion = ContrastiveLoss(margin)
cosine_contrastive_loss_criterion = CosineContrastiveLoss(margin)

# start training
model, optimizer, train_losses, valid_losses = training_loop(model, cross_entropy_loss_criterion,cosine_contrastive_loss_criterion,BATCH_SIZE, optimizer, N_EPOCHS,contrastive_ratio,margin, DEVICE)
 

