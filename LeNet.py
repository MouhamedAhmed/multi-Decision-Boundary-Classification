import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.n = n_classes
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )
        self.classifier2 = nn.Linear(in_features=84, out_features=n_classes)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        l1 = self.classifier1(x)
        l2 = self.classifier2(l1)
        probs = F.softmax(l2, dim=1)
       
        return l1, probs