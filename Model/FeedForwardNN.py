import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the feed-forward neural network.
        Args:
            x: Input tensor of shape (batch_size, embedding_dim).
        returns:
            Tensor of shape (batch_size, embedding_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




