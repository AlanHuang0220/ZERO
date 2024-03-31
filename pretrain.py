import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from Dataset import CustomCollateFn, CustomDataset
from Model import BaselineZERO
from  torch import optim
import torch.nn.functional as F
from Metrics import cosine_similarity_matrix, tensor_text_to_video_metrics, compute_metrics

if __name__ == "__main__":

    # ***** Hyperparameter *****
    epochs = 10
    batch_size = 32
    learning_rate=0.001
    embedding_dim = 512
    num_heads = 2
    ff_hidden_dim = 1024
    learnable_token_num = 10
    dropout = 0
    num_block = 2
    max_seq_len = 200
    # ***** Hyperparameter *****

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    collate_fn = CustomCollateFn(tokenizer)
    dataset_folder = r'D:\Zero\test_data'
    dataset = CustomDataset(dataset_folder, visual_encoder='CLIP', audio_encoder='CLAP')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = BaselineZERO(
        embedding_dim=embedding_dim, 
        num_heads=num_heads, 
        ff_hidden_dim=ff_hidden_dim, 
        learnable_token_num=learnable_token_num, 
        dropout=dropout, 
        num_block=num_block, 
        max_seq_len=max_seq_len
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            # print(batch.keys())
            vision_representation, text_repreaentation, loss = model(batch['vision_feature'], batch['vast_cap'], batch['media_mask'], batch['vast_cap_mask'])
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient for {name}:')
            #         print(param.grad)
            average_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {average_loss:.4f}")

            

        