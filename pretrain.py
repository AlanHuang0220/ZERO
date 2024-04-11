import torch
from torch.utils.data import DataLoader
from Dataset.Dataset import CustomCollateFn, VAST27MDataset
from Model import ModalSpecZERO, BaselineZERO
from torch import optim
from tqdm import tqdm
from Metrics import tensor_text_to_video_metrics, cosine_similarity_matrix


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ***** Hyperparameter *****
    epochs = 1000
    batch_size = 2
    learning_rate=0.001
    embedding_dim = 512
    num_heads = 8
    ff_hidden_dim = 1024
    learnable_token_num = 100
    dropout = 0.1
    num_block = 16
    max_seq_len = 200
    temperature=0.1
    features_to_load = ['CLIP']
    captions_to_load = ['vast_cap']
    # dataset_folder = "/home/miislab-server2/Alan/Alan_shared/VAST27M/video_feature"
    dataset_folder = "./test_data"
    t_max = 100
    best_eval_loss = float('inf')
    # ***** Hyperparameter *****

    dataset = VAST27MDataset(dataset_folder=dataset_folder, features_to_load=features_to_load, captions_to_load=captions_to_load, test_size=0.1, random_state=42)

    collate_fn = CustomCollateFn(tokenizer_model_name='google/flan-t5-xl', use_fast=True)
    tokenizer = collate_fn.get_tokenizer()

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # eval_loader = DataLoader(dataset.eval_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = ModalSpecZERO(
        embedding_dim=embedding_dim, 
        num_heads=num_heads, 
        ff_hidden_dim=ff_hidden_dim, 
        learnable_token_num=learnable_token_num, 
        dropout=dropout, 
        num_block=num_block, 
        max_seq_len=max_seq_len,
        temperature=temperature
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: [bold]{total_params}[/bold]")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # for i, batch in enumerate(train_loader):
    #     if i == 5:
    #         nth_batch = batch
    #         break
    # print(nth_batch['clip_id'])

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            # print(batch.keys())
            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

            vision_learnable_token_output, text_learnable_token_output= model(
                batch['CLIP_feature'].to(device), 
                batch['vast_cap'].to(device), 
                batch['CLIP_feature_mask'].to(device), 
                batch['vast_cap_mask'].to(device)
            )
            # _, loss = model.calculate_lm_loss(batch['vast_cap'].to(device), vision_learnable_token_output, tokenizer)
            # print(type(batch['CLIP_feature']))
            loss = model.calculate_mlm_loss(batch['CLIP_feature'].to(device), batch['CLIP_feature_mask'].to(device), text_learnable_token_output, mask_ratio=0.15)
            # print(loss.item())
            # masked_feature, mask = model.random_feature_masking(batch['CLIP_feature'].to(device))
            # print(masked_feature)
            # print(mask)
            # print(loss)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient for {name}:')
            #         print(param.grad)
        lr_scheduler.step()
        average_loss = total_train_loss / len(train_loader)
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] - Loss: {average_loss:.4f}")
        
        # model.eval()
        # total_eval_loss = 0.0

        # with torch.no_grad():
        #     # for batch in train_loader:
        #     vision_representation, text_representation = model(
        #         nth_batch['CLIP_feature'].to(device), 
        #         nth_batch['vast_cap'].to(device), 
        #         nth_batch['CLIP_feature_mask'].to(device), 
        #         nth_batch['vast_cap_mask'].to(device)
        #     )
            # total_eval_loss += loss.item()
            
        #     text_output, loss = model.calculate_lm_loss(nth_batch['vast_cap'].to(device), vision_representation, tokenizer)
        #     _, text_output_ids = torch.max(text_output, dim=-1)
        #     result = tokenizer.decode(text_output_ids[0])
        #     tqdm.write(result)


            # cos_sim = cosine_similarity_matrix(vision_representation[:,1,:], text_representation[:,1,:]).unsqueeze(0)
            # # print(cos_sim)
            # result = tensor_text_to_video_metrics(cos_sim.cpu())
            # print(result)

            
            # average_eval_loss = total_eval_loss / len(train_loader)
            # tqdm.write(f"Eval Loss: {average_eval_loss:.4f}")

        # if average_eval_loss < best_eval_loss:
        #     best_eval_loss = average_eval_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     tqdm.write("Saved Best Model!")


        