import Model
import Dataset.Dataset as Dataset 
import torch.utils.data
import torch
import wandb
import os
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from utils import load_config, display_config, initialize_component
from  torch import optim
from rich import print

if __name__ == "__main__":
    config = load_config('config.yaml')
    wandb.init(config=config, **config['experiment']['args'])
    display_config(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: [bold green]{device}")

    dataset = initialize_component(config['dataset'], Dataset, config['dataset']['args'])

    collate_fn = initialize_component(config['collate_fn'], Dataset, config['collate_fn']['args'])
    tokenizer = collate_fn.get_tokenizer()

    config['train_loader']['args']['dataset'] = dataset.train_subset
    config['train_loader']['args']['collate_fn'] = collate_fn
    train_loader = initialize_component(config['train_loader'], torch.utils.data, config['train_loader']['args'])

    config['eval_loader']['args']['dataset'] = dataset.eval_subset
    config['eval_loader']['args']['collate_fn'] = collate_fn
    eval_loader = initialize_component(config['eval_loader'], torch.utils.data, config['eval_loader']['args'])

    model = initialize_component(config['model'], Model, config['model']['args']).to(device)
    wandb.watch(model, log='all')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: [bold]{total_params}[/bold]")

    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training_params']['t_max'])

    model_save_path = os.path.join(config['training_params']['model_save_root_path'], config['experiment']['args']['name'])
    os.makedirs(model_save_path)
    best_eval_loss = float('inf')

    epochs = config['training_params']['epochs']
    with Progress(
            TextColumn("[bold green]{task.description}", justify="left"),
            BarColumn(bar_width=40),
            TextColumn("[bold][progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("[bold red]{task.completed}/{task.total}"),
            TextColumn("[bold cyan]{task.fields[info]}"),
            transient=True
        ) as progress:
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            
            training_task = progress.add_task("Epoch [{}/{}]".format(epoch+1, epochs), total=len(train_loader), info="N/A")

            for batch_idx, batch in enumerate(train_loader):
                # print(batch.keys())
                vis_feat = batch['CLIP_feature'].to(device, non_blocking=True)
                text_ids = batch['vision_cap'].to(device, non_blocking=True)
                vis_pad_mask = batch['CLIP_feature_mask'].to(device, non_blocking=True)
                text_pad_mask = batch['vision_cap_mask'].to(device, non_blocking=True)

                vision_representation, text_representation = model(vis_feat, text_ids, vis_pad_mask, text_pad_mask)

                loss = model.get_loss(vis_feat, text_ids, vis_pad_mask, vision_representation, text_representation, tokenizer=tokenizer)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                progress.update(training_task, advance=1, info=f"Current Loss: {loss.item():.4f}")

            lr_scheduler.step()
            average_loss = total_train_loss / len(train_loader)
            # progress.update(training_task, info=f"Train Avg Loss: {average_loss:.4f}")
            wandb.log({"Train Loss": average_loss}, step=epoch+1)
            wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=epoch+1)

            model.eval()
            total_eval_loss = 0.0
            eval_task = progress.add_task(f"[magenta]Evaluating Epoch [{epoch+1}/{epochs}]", total=len(eval_loader), info="N/A")

            with torch.no_grad():
                for batch in eval_loader:
                    vis_feat = batch['CLIP_feature'].to(device, non_blocking=True)
                    text_ids = batch['vision_cap'].to(device, non_blocking=True)
                    vis_pad_mask = batch['CLIP_feature_mask'].to(device, non_blocking=True)
                    text_pad_mask = batch['vision_cap_mask'].to(device, non_blocking=True)

                    vision_representation, text_representation = model(vis_feat, text_ids, vis_pad_mask, text_pad_mask)

                    loss = model.get_loss(vis_feat, text_ids, vis_pad_mask, vision_representation, text_representation, tokenizer=tokenizer)

                    total_eval_loss += loss.item()
                    progress.update(eval_task, advance=1, info=f"Current Loss: {loss.item():.4f}")
                
            average_eval_loss = total_eval_loss / len(eval_loader)
            progress.remove_task(eval_task)
            progress.remove_task(training_task)
            # progress.update(training_task, info=f"Train Avg Loss: {average_loss:.4f}, Eval Avg Loss: {average_eval_loss:.4f}")
            progress.log(f"[bold green]Epoch [{epoch+1}/{epochs}] - Train Avg Loss: {average_loss:.4f} - Eval Avg Loss: {average_eval_loss:.4f}")
            wandb.log({"Eval Loss": average_eval_loss}, step=epoch+1)
            
            # 保存eval loss最低的模型
            if average_eval_loss < best_eval_loss:
                best_eval_loss = average_eval_loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

            # 每10個epoch保存一個模型
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth'))
