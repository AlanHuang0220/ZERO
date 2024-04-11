import torch.nn as nn
from .FeedForwardNN import FeedForwardNN

class ModalAgnZEROformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(ModalAgnZEROformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForwardNN(embedding_dim, ff_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, self_attn_mask, cross_attn_mask):
        # Self Attention
        attention_output, _ = self.self_attention(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Cross Attention
        attention_output, _ = self.cross_attention(x, context, context, attn_mask=cross_attn_mask)
        x = x + self.dropout(attention_output)
        x = self.norm2(x)

        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x
    
class ModalSpecZEROformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(ModalSpecZEROformerBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.feed_forward_1 = FeedForwardNN(embedding_dim, ff_hidden_dim)
        self.feed_forward_2 = FeedForwardNN(embedding_dim, ff_hidden_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, self_attn_mask, cross_attn_mask, use_mode):
        # Self Attention
        attention_output, _ = self.self_attention(x, x, x, attn_mask=self_attn_mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Cross Attention
        attention_output, _ = self.cross_attention(x, context, context, attn_mask=cross_attn_mask)
        x = x + self.dropout(attention_output)
        x = self.norm2(x)

        if use_mode == 'text':
            ff_output = self.feed_forward_1(x)
            x = x + self.dropout(ff_output)
            x = self.norm3(x)
        elif use_mode == 'vision':
            ff_output = self.feed_forward_2(x)
            x = x + self.dropout(ff_output)
            x = self.norm4(x)
        else:
            raise ValueError('use_mode should be either text or vision')

        return x