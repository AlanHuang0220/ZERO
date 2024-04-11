import torch.nn as nn
import torch

class AddPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(AddPositionalEmbedding, self).__init__()
        # Initialize positional embeddings as a learnable parameter
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))

        nn.init.normal_(self.position_embedding, mean=0, std=embedding_dim ** -0.5)

    def forward(self, input):
        """
        Adds positional embeddings to input tensor.
        Parameters:
            input: Tensor of size [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor with added positional embeddings.
        """
        seq_len = input.size(1)
        # Slice the positional embeddings to match the sequence length of the input
        pos_embeddings = self.position_embedding[:seq_len, :]
        # Add positional embeddings to the input
        return input + pos_embeddings