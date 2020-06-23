import torch.nn.functional as F
import torch
import torch.nn as nn

class LossLayer(nn.Module):

    def __init__(self, device):
        super(LossLayer, self).__init__()
        weights = torch.Tensor(1, 2)
        weights.fill_(1.0)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        self.device = device
       
    def forward(self, contrastive_loss, cross_entropy_loss, contrastive_ratio):
        x = self.weights[0][0]
        y = self.weights[0][1]
        x = x.to(self.device)
        y = y.to(self.device)
        contrastive_loss = contrastive_loss.to(self.device)
        cross_entropy_loss = cross_entropy_loss.to(self.device)
        loss = (contrastive_ratio * torch.pow(contrastive_loss,x)) + ((1 - contrastive_ratio) * torch.pow(cross_entropy_loss,y))
        return loss

