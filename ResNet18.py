import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from torchvision.models.resnet import resnet18
class ResNet18(nn.Module):

    def __init__(self, n_classes):
        super(ResNet18, self).__init__()
        self.n = n_classes
        self.feature_extractor = resnet18()   
        self.classifier = nn.Linear(in_features=1000, out_features=n_classes)



    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
       
        return x, probs