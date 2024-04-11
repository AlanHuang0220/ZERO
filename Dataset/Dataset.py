import numpy as np
import random
import pickle
import torch
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer
from utils import check_memory
from typing import List
from tqdm import tqdm

class VAST27MDataset(Dataset):
    def __init__(self, dataset_folder, features_to_load: List[str], captions_to_load: List[str], test_size, random_state):
        self.dataset_folder = dataset_folder
        self.clip_ids = os.listdir(dataset_folder)
        self.features_to_load = features_to_load
        self.captions_to_load = captions_to_load

        data_indices = list(range(len(self.clip_ids)))
        train_idx, eval_idx = train_test_split(data_indices, test_size=test_size, random_state=random_state)
        self.train_subset = Subset(self, train_idx)
        self.eval_subset = Subset(self, eval_idx)

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        data_point = {}

        for encode_model_name in self.features_to_load:
            feature_file_path = os.path.join(self.dataset_folder, clip_id, f"{encode_model_name}.pkl")
            with open(feature_file_path, 'rb') as f:
                feature = pickle.load(f)
                data_point[f'{encode_model_name}_feature'] = feature

        caption_file_path = os.path.join(self.dataset_folder, clip_id, "caption.json")
        with open(caption_file_path, 'r') as f:
            metadata = json.load(f)
            for caption in self.captions_to_load:
                if caption in metadata:
                    data_point[caption] = metadata[caption]
        data_point['clip_id'] = clip_id
        return data_point
    
class VAST27MDatasetPreload(Dataset):
    def __init__(self, dataset_folder, features_to_load: List[str], captions_to_load: List[str], test_size, random_state):
        self.data = []
        clip_ids = os.listdir(dataset_folder)

        for clip_id in tqdm(clip_ids, total=len(clip_ids)):
            if check_memory():
                print("memory 不夠啦 !!!")
                break

            data_point = {}
            for encode_model_name in features_to_load:
                feature_file_path = os.path.join(dataset_folder, clip_id, f"{encode_model_name}.pkl")
                with open(feature_file_path, 'rb') as f:
                    feature = pickle.load(f)
                    data_point[f'{encode_model_name}_feature'] = feature

            caption_file_path = os.path.join(dataset_folder, clip_id, "caption.json")
            with open(caption_file_path, 'r') as f:
                metadata = json.load(f)
                for caption in captions_to_load:
                    if caption in metadata:
                        data_point[caption] = metadata[caption]

            self.data.append(data_point)
    
        train_idx, eval_idx = train_test_split(list(range(len(self.data))), test_size=test_size, random_state=random_state)
        self.train_subset = Subset(self, train_idx)
        self.eval_subset = Subset(self, eval_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class CustomCollateFn:
    def __init__(self, tokenizer_model_name, use_fast):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=use_fast)
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    def __call__(self, batch):
        collated_batch = {}
        for key in list(batch[0].keys()):
            if key.endswith('feature'):
                features = [torch.from_numpy(item[key]) for item in batch]
                features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
                collated_batch[key] = features_padded
                collated_batch[f'{key}_mask'] = torch.from_numpy(np.where(np.all(collated_batch[key].numpy() == 0, axis=2), True, False))

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
            
            else :
                collated_batch[key] = [item[key] for item in batch]

        return collated_batch
    
    def get_tokenizer(self):
        return self.tokenizer

