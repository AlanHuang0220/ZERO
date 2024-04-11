import torch
import torch.nn as nn
from Loss.BilateralInfoNCELoss import BilateralInfoNCELoss
from .PositionEmb import AddPositionalEmbedding
from .ZEROFormerBlock import ModalSpecZEROformerBlock

class ModalSpecZERO(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_hidden_dim, learnable_token_num, dropout, num_block, max_seq_len=200, temperature=0.1, loss_configs=None):
        super(ModalSpecZERO, self).__init__()
        self.num_heads = num_heads
        self.learnable_token_num = learnable_token_num
        self.text_learnable_token = nn.Parameter(torch.randn(1, learnable_token_num, embedding_dim))
        self.vision_learnable_token = nn.Parameter(torch.randn(1, learnable_token_num, embedding_dim))

        self.embedding_layer = nn.Embedding(num_embeddings=32101, embedding_dim=embedding_dim)
        self.add_position_emb = AddPositionalEmbedding(max_seq_len+1, embedding_dim)

        self.zeroformer_blocks = nn.ModuleList()
        for _ in range(num_block):
            self.zeroformer_blocks.append(ModalSpecZEROformerBlock(embedding_dim, num_heads, ff_hidden_dim, dropout))
        
        self.lm_head = nn.Linear(embedding_dim, 32101)

        self.bilateral_info_nce_fn = BilateralInfoNCELoss(temperature)
        self.cross_entropy_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.mse_fn = nn.L1Loss(reduction='sum')

        self.loss_configs = loss_configs if loss_configs is not None else []

    def forward(self, vis_feat, text_ids, vis_pad_mask, text_pad_mask):
        vision_learnable_tokens = self.vision_learnable_token.expand(vis_feat.size(0), -1, -1)
        text_learnable_tokens = self.text_learnable_token.expand(vis_feat.size(0), -1, -1)

        vision_emb = self.add_position_emb(vis_feat)
        text_emb = self.add_position_emb(self.embedding_layer(text_ids))

        text_pad_mask = text_pad_mask.unsqueeze(1).expand(-1, self.learnable_token_num, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        text_pad_mask = text_pad_mask.reshape(text_pad_mask.size(0) * self.num_heads, text_pad_mask.size(2), text_pad_mask.size(3))

        vis_pad_mask = vis_pad_mask.unsqueeze(1).expand(-1, self.learnable_token_num, -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        vis_pad_mask = vis_pad_mask.reshape(vis_pad_mask.size(0) * self.num_heads, vis_pad_mask.size(2), vis_pad_mask.size(3))
        
        for zeroformer_block in self.zeroformer_blocks:
            text_learnable_tokens = zeroformer_block(text_learnable_tokens, text_emb, self_attn_mask=None, cross_attn_mask=text_pad_mask, use_mode='text')
            vision_learnable_tokens = zeroformer_block(vision_learnable_tokens, vision_emb, self_attn_mask=None, cross_attn_mask=vis_pad_mask, use_mode='vision')

        return vision_learnable_tokens, text_learnable_tokens
    
    def _calculate_info_nce_loss(self, vision_learnable_tokens, text_learnable_tokens):
        bilateral_info_nce_loss = self.bilateral_info_nce_fn(vision_learnable_tokens[:, 1, :], text_learnable_tokens[:, 1, :])
        return bilateral_info_nce_loss
    
    def _calculate_lm_loss(self, text_ids, vision_learnable_tokens, tokenizer):
        bos_token_tensor = torch.full((text_ids.size(0), 1), tokenizer.bos_token_id).to(text_ids.device) # create bos token tensor with shape (batch_size, 1)  
        text_input = torch.cat([bos_token_tensor, text_ids[:, :-1]], dim=1) # concatenate bos token and text input
        text_input = self.add_position_emb(self.embedding_layer(text_input)) # embed text input and add positional embeddings

        causal_mask = torch.triu(torch.ones((text_ids.size(1), text_ids.size(1))), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(text_ids.size(0)*self.num_heads, -1, -1).to(text_ids.device)

        for zeroformer_block in self.zeroformer_blocks:
            text_output = zeroformer_block(text_input, vision_learnable_tokens, self_attn_mask=causal_mask, cross_attn_mask=None, use_mode='text')

        text_output = self.lm_head(text_output)
        lm_loss = self.cross_entropy_fn(text_output.view(-1, text_output.size(-1)), text_ids.view(-1))
        return text_output, lm_loss
    
    def _random_feature_masking(self, features, mask_ratio=0.15, mask_value=0):
        """
        randomly mask tokens of the input tensor.

        args:
        features (Tensor): input tensor
        mask_ratio (float): ratio of masked tokens
        mask_value (int): value to be filled in masked positions
        returns:
        masked_features (Tensor): masked tensor
        """
        masked_features = features.clone() 
        
        batch_size, seq_len, _ = features.shape
        num_mask = int(mask_ratio * seq_len)

        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        # mask the input tensor with random indices for each batch
        for i in range(batch_size):
            mask_indices = torch.randperm(seq_len)[:num_mask]
            masked_features[i, mask_indices, :] = mask_value
            mask[i, mask_indices] = True
        
        return masked_features, mask.to(features.device)

    def _calculate_mlm_loss(self, vis_feat, vis_pad_mask, text_learnable_tokens, mask_ratio=0.15, mask_value=0):
        masked_vis_feat, masked_vis_mask = self._random_feature_masking(vis_feat, mask_ratio=mask_ratio, mask_value=mask_value)
        masked_vis_feat = self.add_position_emb(masked_vis_feat) # add positional embeddings

        vis_self_attn_mask = vis_pad_mask.unsqueeze(1).expand(-1, vis_feat.size(1), -1).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        vis_self_attn_mask = vis_self_attn_mask.reshape(vis_self_attn_mask.size(0) * self.num_heads, vis_self_attn_mask.size(2), vis_self_attn_mask.size(3))
       

        for zeroformer_block in self.zeroformer_blocks:
            masked_vis_feat = zeroformer_block(masked_vis_feat, text_learnable_tokens, self_attn_mask=vis_self_attn_mask, cross_attn_mask=None, use_mode='vision')
        
        masked_vis_feat = masked_vis_feat.masked_fill(vis_pad_mask.unsqueeze(-1), 0)

        loss_denominator = torch.sum(~vis_pad_mask & masked_vis_mask)
        mlm_loss = self.mse_fn(masked_vis_feat[masked_vis_mask], vis_feat[masked_vis_mask]) / loss_denominator
        return mlm_loss

    def get_loss(self, vis_feat, text_ids, vis_pad_mask, vision_learnable_tokens, text_learnable_tokens, tokenizer=None):
        total_loss = 0
        for loss_config in self.loss_configs:
            loss_type = loss_config['type']
            weight = loss_config.get('weight', 1)
            
            if loss_type == 'info_nce':
                loss = self._calculate_info_nce_loss(vision_learnable_tokens, text_learnable_tokens)
            elif loss_type == 'lm_loss':
                assert tokenizer is not None, "Tokenizer must be provided for lm_loss calculation"
                _, loss = self._calculate_lm_loss(text_ids, vision_learnable_tokens, tokenizer)
            elif loss_type == 'mlm_loss':
                loss = self._calculate_mlm_loss(vis_feat, vis_pad_mask, text_learnable_tokens, mask_ratio=loss_config['mask_ratio'])
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            total_loss += weight * loss
    
        return total_loss