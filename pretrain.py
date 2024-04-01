import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from Dataset import CustomCollateFn, VAST27MDataset
from Model import BaselineZERO
from  torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from Metrics import cosine_similarity_matrix, tensor_text_to_video_metrics, compute_metrics
import wandb

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
    temperature=0.1
    features_to_load = ['CLIP']
    captions_to_load = ['vision_cap']
    dataset_folder = r'D:\Zero\test_data'
    shuffle = False
    t_max = 100
    # ***** Hyperparameter *****

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    collate_fn = CustomCollateFn(tokenizer)

    full_dataset = VAST27MDataset(dataset_folder=dataset_folder, features_to_load=features_to_load, captions_to_load=captions_to_load)
    train_idx, eval_idx = train_test_split(list(range(len(full_dataset))), test_size=0.1, random_state=42)

    # 創建訓練集和評估集的Subset
    train_subset = Subset(full_dataset, train_idx)
    eval_subset = Subset(full_dataset, eval_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_subset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = BaselineZERO(
        embedding_dim=embedding_dim, 
        num_heads=num_heads, 
        ff_hidden_dim=ff_hidden_dim, 
        learnable_token_num=learnable_token_num, 
        dropout=dropout, 
        num_block=num_block, 
        max_seq_len=max_seq_len,
        temperature=temperature
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # 初始化 Weights & Biases
    wandb.init(project='ZERO', entity='alan0220', name='BaselineZERO_experiment')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            vision_representation, text_repreaentation, loss = model(
                batch['CLIP_feature'], batch['vision_cap'], batch['CLIP_feature_mask'], batch['vision_cap_mask']
            )
            # vision_cls, text_cls = vision_representation[:, 1, :], text_repreaentation[:, 1, :]
            # sim_matrix = cosine_similarity_matrix(vision_cls, text_cls).unsqueeze(0)
            # results = tensor_text_to_video_metrics(sim_matrix)
            # formatted_results = {k: "{:.2f}".format(v) for k, v in results.items()}
            # print(formatted_results)

            # # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient for {name}:')
            #         print(param.grad)
            average_loss = total_train_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {average_loss:.4f}")
            wandb.log({"Train Loss": average_loss}, step=epoch)
        
        model.eval()
        total_eval_loss = 0.0

        with torch.no_grad():  # 關閉梯度
            for batch in eval_loader:
                vision_representation, text_representation, loss = model(
                    batch['CLIP_feature'], batch['vision_cap'], batch['CLIP_feature_mask'], batch['vision_cap_mask']
                )
                
                total_eval_loss += loss.item()
            
            average_eval_loss = total_eval_loss / len(eval_loader)
            print(f"Eval Loss: {average_eval_loss:.4f}")
            wandb.log({"Eval Loss": average_eval_loss}, step=epoch)
            

            vision_cls, text_cls = vision_representation[:, 1, :], text_repreaentation[:, 1, :]
            sim_matrix = cosine_similarity_matrix(vision_cls, text_cls).unsqueeze(0)
            eval_results = tensor_text_to_video_metrics(sim_matrix)
            formatted_eval_results = {k: "{:.2f}".format(v) for k, v in eval_results.items()}
            print("Evaluation Metrics:", formatted_eval_results)
            wandb.log({f"metric/{key}": value for key, value in eval_results.items()}, step=epoch)
            

        