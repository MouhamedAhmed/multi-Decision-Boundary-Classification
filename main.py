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
from ResNet18 import *
from CustomNet import *
from train_val import *
from torch.utils.data.sampler import SubsetRandomSampler
from contrastive_loss import *
from cosine_contrastive_loss import *
from mul_cosine_contrastive_loss import *
from loss_layer import *

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
N_EPOCHS = 150

IMG_SIZE = 224
N_CLASSES = 55
margin = 32
m1 = 1.2
m2 = 0.0
print('contrastive ratio:')
contrastive_ratio = float(input())
# contrastive_ratio = 0.5

# instantiate the model
torch.manual_seed(RANDOM_SEED)

model = ResNet18(N_CLASSES).to(DEVICE)
# in case of resnet
# model.feature_extractor.fc = nn.Linear(512,N_CLASSES)
model.feature_extractor[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cross_entropy_loss_criterion = nn.CrossEntropyLoss()
contrastive_loss_criterion = ContrastiveLoss(margin)
cosine_contrastive_loss_criterion = CosineContrastiveLoss(margin)
mul_cosine_contrastive_loss_criterion = MulCosineContrastiveLoss(margin,m1,m2)
lossLayer = LossLayer(DEVICE)
# start training
model, optimizer, train_losses, valid_losses = training_loop(model, cross_entropy_loss_criterion,cosine_contrastive_loss_criterion,lossLayer,BATCH_SIZE, optimizer, N_EPOCHS,contrastive_ratio,margin, DEVICE)
 

