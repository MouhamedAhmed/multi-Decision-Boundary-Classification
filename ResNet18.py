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
        self.feature_extractor = torch.nn.Sequential(*(list(resnet18().children())[:-1])) 
        self.classifier = nn.Linear(in_features=512, out_features=n_classes)

        self.classifier1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=n_classes),
            nn.ReLU(),
        )



    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # logits = self.classifier1(x)
        probs = F.softmax(logits, dim=1)
       
        return x, probs

# m = ResNet18(55)   
# print(m)
# i = np.random.rand(16,3,128,128)
# t = torch.from_numpy(i).to('cuda')
# # t = t.type(torch.DoubleTensor)
# # print(t)
# m = m.float().to('cuda')
# u,p = m(t.float())
# print(u.size(),p.size())
# layer = m.feature_extractor._modules.get('avgpool')

# newmodel = torch.nn.Sequential(*(list(resnet18().children())[:-1]))
# print(newmodel)