class CosineContrastiveLoss(nn.Module):
    """
    Cosine contrastive loss function.
    Based on: http://anthology.aclweb.org/W16-1617
    Maintain 0 for match, 1 for not match.
    If they match, loss is 1/4(1-cos_sim)^2.
    If they don't, it's cos_sim^2 if cos_sim < margin or 0 otherwise.
    Margin in the paper is ~0.4.
    """

    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con