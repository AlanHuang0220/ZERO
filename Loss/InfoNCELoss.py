import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        計算InfoNCE損失
        :param features1: 第一個模態的特徵，形狀應為 (batch_size, feature_dim)
        :param features2: 第二個模態的特徵，形狀應為 (batch_size, feature_dim)
        :return: InfoNCE損失
        """
        # 標準化特徵向量
        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        # 計算相似度矩陣
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature

        # 計算損失
        batch_size = features1.size(0)
        labels = torch.arange(batch_size).to(features1.device)
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
    
