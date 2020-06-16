import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

class CustomNet(nn.Module):

    def __init__(self, n_classes):
        super(CustomNet, self).__init__()
        self.n = n_classes
        self.feature_extractor1 = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.feature_extractor2 = nn.Sequential(            
           
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.feature_extractor3 = nn.Sequential(           
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

        )

        self.feature_extractor4 = nn.Sequential(           

            nn.Conv2d(in_channels=32, out_channels=120, kernel_size=5, stride=1),
            nn.BatchNorm2d(120),
            nn.ReLU(),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU()
        )
        self.classifier2 = nn.Linear(in_features=84, out_features=n_classes)
        self.sig = nn.Sigmoid()



    def forward(self, x):
        x = self.feature_extractor1(x)
        print(x.size())
        x = self.feature_extractor2(x)
        print(x.size())
        x = self.feature_extractor3(x)
        print(x.size())
        x = self.feature_extractor4(x)
        print(x.size())
        x = torch.flatten(x, 1)
        l1 = self.classifier1(x)
        l2 = self.classifier2(l1)
        probs = F.softmax(l2, dim=1)
       
        return l1, probs