import torch
import torch.nn as nn
from Loss.BilateralInfoNCELoss import BilateralInfoNCELoss
from .PositionEmb import AddPositionalEmbedding
from .ZEROFormerBlock import ModalAgnZEROformerBlock

class BaselineZERO(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, learnable_token_num, dropout, num_block, max_seq_len=200, temperature=0.1):
        super(BaselineZERO, self).__init__()
        self.num_heads = num_heads
        self.learnable_token_num = learnable_token_num
        self.text_learnable_token = nn.Parameter(torch.randn(1, learnable_token_num, embedding_dim))
        self.vision_learnable_token = nn.Parameter(torch.randn(1, learnable_token_num, embedding_dim))

        self.embedding_layer = nn.Embedding(num_embeddings=32100, embedding_dim=embedding_dim)
        self.add_position_emb = AddPositionalEmbedding(max_seq_len+1, embedding_dim)

        self.zeroformer_blocks = nn.ModuleList()
        for _ in range(num_block):
            self.zeroformer_blocks.append(ModalAgnZEROformerBlock(embedding_dim, num_heads, ff_hidden_dim, dropout))
        
        self.loss_calculator = BilateralInfoNCELoss(temperature)

    def forward(self, vision_feature, text_ids, vision_mask, text_mask):
        vision_learnable_tokens = self.vision_learnable_token.expand(vision_feature.size(0), -1, -1)
        text_learnable_tokens = self.text_learnable_token.expand(vision_feature.size(0), -1, -1)

        vision_emb = self.add_position_emb(vision_feature)
        text_emb = self.add_position_emb(self.embedding_layer(text_ids))

        text_mask = text_mask.unsqueeze(1).expand(-1, self.learnable_token_num, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        text_mask = text_mask.reshape(text_mask.size(0) * self.num_heads, text_mask.size(2), text_mask.size(3))

        vision_mask = vision_mask.unsqueeze(1).expand(-1, self.learnable_token_num, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        vision_mask = vision_mask.reshape(vision_mask.size(0) * self.num_heads, vision_mask.size(2), vision_mask.size(3))
        
        for zeroformer_block in self.zeroformer_blocks:
            vision_learnable_tokens = zeroformer_block(vision_learnable_tokens, vision_emb, self_attn_mask=None, cross_attn_mask=vision_mask)
            text_learnable_tokens = zeroformer_block(text_learnable_tokens, text_emb, self_attn_mask=None, cross_attn_mask=text_mask)

        # print(vision_learnable_tokens[:, 1, :])
        return vision_learnable_tokens, text_learnable_tokens
    
    def calculate_info_nce_loss(self, vision_learnable_tokens, text_learnable_tokens):
        return self.loss_calculator(vision_learnable_tokens[:, 1, :], text_learnable_tokens[:, 1, :])




