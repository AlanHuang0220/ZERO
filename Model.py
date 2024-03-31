import torch
import torch.nn as nn
import torch.nn.functional as F
from Loss import BilateralInfoNCELoss

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
    
class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ZEROformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(ZEROformerBlock, self).__init__()
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
            self.zeroformer_blocks.append(ZEROformerBlock(embedding_dim, num_heads, ff_hidden_dim, dropout))
        
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
        loss = self.loss_calculator(vision_learnable_tokens[:, 1, :], text_learnable_tokens[:, 1, :])
        return vision_learnable_tokens, text_learnable_tokens, loss



# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer
# from Dataset import CustomCollateFn, CustomDataset

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
# collate_fn = CustomCollateFn(tokenizer)
# dataset_folder = r'D:\Zero\test_data'
# dataset = CustomDataset(dataset_folder, visual_encoder='CLIP', audio_encoder='CLAP')
# dataloader = DataLoader(dataset, batch_size=15, shuffle=False, collate_fn=collate_fn)

# model = BaselineZERO(embedding_dim=512, num_heads=2, ff_hidden_dim=1024, learnable_token_num=10, dropout=0, num_block=2, max_seq_len=200)

# for batch in dataloader:
#     print(batch.keys())
#     # print(batch['vision_cap_mask'])
#     vision_feature = batch['vision_feature']
#     text_ids = batch['vision_cap']
#     vision_mask = batch['media_mask']
#     text_mask = batch['vision_cap_mask']
#     vision_representation, audio_repreaentation, loss = model(vision_feature, text_ids, vision_mask, text_mask)
#     print(loss)
#     # output = model()

