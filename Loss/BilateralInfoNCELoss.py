import torch.nn as nn
from .InfoNCELoss import InfoNCELoss

class BilateralInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(BilateralInfoNCELoss, self).__init__()
        self.info_nce = InfoNCELoss(temperature)
    
    def forward(self, features1, features2):
        return (self.info_nce(features1, features2) + self.info_nce(features2, features1))/2