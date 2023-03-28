import torch
from torch.utils.data import Dataset

class ValDataset(Dataset):
    def __init__(self, tokenizer):
      self.ids = tokenizer['input_ids']
      self.attn_mask = tokenizer['attention_mask']
      self.t_type_ids = tokenizer['token_type_ids']

    def __len__(self):
      return len(self.ids)
    
    def __getitem__(self, idx):
      return self.ids[idx], self.attn_mask[idx], self.t_type_ids[idx]