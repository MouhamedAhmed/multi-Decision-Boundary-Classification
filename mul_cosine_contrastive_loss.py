import torch.nn.functional as F
import torch
import torch.nn as nn

class MulCosineContrastiveLoss(nn.Module):
    """
    Cosine contrastive loss function.
    Based on: http://anthology.aclweb.org/W16-1617
    Maintain 0 for match, 1 for not match.
    If they match, loss is 1/4(1-cos_sim)^2.
    If they don't, it's cos_sim^2 if cos_sim < margin or 0 otherwise.
    Margin in the paper is ~0.4.
    """

    def __init__(self, margin = 0.4,m1 = 0, m2 = 0):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        output1_norm = output1 / output1.norm(dim=1)[:, None]
        output2_norm = output2 / output2.norm(dim=1)[:, None]
        # cos_sim = torch.mm(output1_norm, output2_norm.transpose(0,1))
        cos_sim = torch.bmm(output1_norm.view(output1_norm.size()[0], 1, output1_norm.size()[1]), output2_norm.view(output2_norm.size()[0], output2_norm.size()[1], 1))
        # print("output1:",output1.size())
        # print("output2:",output2.size())
        # print("output1_norm:",output1_norm.size())
        # print("output1_norm:",output2_norm.size())
        # print("cos_sim:",cos_sim.size())
        # output1_mag = torch.norm(output1)
        # output2_mag = torch.norm(output2)
        # output1_norm = torch.div(output1,output1_mag)
        # output2_norm = torch.div(output2,output2_mag)
        # cos_sim = torch.mm(output1_norm,output2_norm)

        # cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin).float(), 2))

        loss_theta = torch.acos(loss_cos_con)
        loss_cos_con = torch.cos(loss_theta * m1 + m2)
        
        # print("loss:",loss_cos_con)
        return loss_cos_con