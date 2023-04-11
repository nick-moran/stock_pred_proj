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
    

class LMDataset(Dataset):
    def __init__(self, info_tensor, labels):
      self.ids = info_tensor[:,0,:]
      self.attn_mask = info_tensor[:,1,:]
      self.t_type_ids = info_tensor[:,2,:]
      self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
      return len(self.labels)
    
    def __getitem__(self, idx):
      return self.ids[idx], self.attn_mask[idx], self.t_type_ids[idx], self.labels[idx]