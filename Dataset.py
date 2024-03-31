import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from transformers import AutoTokenizer
import random
from typing import List

class CustomDataset(Dataset):
    def __init__(self, dataset_folder, encoders: List[str], device=None):
        self.dataset_folder = dataset_folder
        self.clip_ids = os.listdir(dataset_folder)
        self.encoders = encoders
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        data = {}
        for encoder in self.encoders:
            feature_file_path = os.path.join(self.dataset_folder, self.clip_ids[idx], f"{encoder}.pkl")
            with open(feature_file_path, 'rb') as f:
                    feature = pickle.load(f)
                    data[f'{encoder}_feature'] = feature.to(self.device)

        caption_file_path = os.path.join(self.dataset_folder, self.clip_ids[idx], "caption.json")
        with open(caption_file_path, 'r') as f:
            metadata = json.load(f)
            data.update(metadata)
        return data
    
# def collate_fn(batch):
#     collated_batch = {}
#     for key in list(batch[0].keys()):
#         if key.endswith('feature'):
#             features = [torch.from_numpy(item[key]) for item in batch]
#             features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
#             collated_batch[key] = features_padded
#         elif key in ['vision_cap', 'audio_cap', 'subtitle', 'vast_cap']:


#     media_mask = np.where(np.all(collated_batch['visual_feature'].numpy() == 0, axis=2), True, False)
#     media_mask = torch.from_numpy(media_mask)
#     collated_batch['media_mask'] = media_mask

#     return collated_batch

class CustomCollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        collated_batch = {}
        for key in list(batch[0].keys()):
            if key.endswith('feature'):
                features = [torch.from_numpy(item[key]) for item in batch]
                features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
                collated_batch[key] = features_padded
                collated_batch[f'{key}_mask'] = torch.from_numpy(np.where(np.all(collated_batch['vision_feature'].numpy() == 0, axis=2), True, False))

            elif key in ['vision_cap', 'audio_cap']:
                choise_caption = [item[key][random.randint(0, len(item[key]) - 1)] for item in batch] #從多個caption中隨機選擇一個
                tokenized_caption = self.tokenizer(choise_caption, return_tensors="pt", padding=True)
                collated_batch[key] = tokenized_caption.input_ids
                collated_batch[f"{key}_mask"] = ~tokenized_caption.attention_mask.bool()

            elif key in ['vast_cap', 'subtitle']:
                caption = [item[key] for item in batch] 
                tokenized_caption = self.tokenizer(caption, return_tensors="pt", padding=True)
                collated_batch[key] = tokenized_caption.input_ids
                collated_batch[f"{key}_mask"] = ~tokenized_caption.attention_mask.bool()

        return collated_batch

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
# collate_fn = CustomCollateFn(tokenizer)
# dataset_folder = r'F:\dataset\pretrain_dataset\VAST27M\video_feature'
# dataset = CustomDataset(dataset_folder, visual_encoder='CLIP', audio_encoder='CLAP')
# dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
# for batch in dataloader:
#     print(batch.keys())
#     print(batch['vast_cap'].shape)

# path = r'D:\Zero\test_data\-_0GokQxPz8.8\CLIP.pkl'
# with open(path, 'rb') as f:
#     data = pickle.load(f)
# print(data)
    
