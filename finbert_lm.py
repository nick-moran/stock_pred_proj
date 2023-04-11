
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloaders.finbert_dataloaders import LMDataset
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import numpy as np
import sys

news_df = pd.read_csv("./finbert_datasets/hourly_apple.csv")
# news_df.drop_duplicates(subset=['text'])
labels = news_df['dir'].to_numpy()
labels[labels < 0] = 0

text = news_df['text'].apply(lambda x: x[:600]).tolist()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# keys = ['input_ids', 'token_type_ids', 'attention_mask']
tokens_encoding = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

ids = tokens_encoding['input_ids']
attn_mask = tokens_encoding['attention_mask']
ttype_ids = tokens_encoding['token_type_ids']

comb = torch.stack((ids,attn_mask,ttype_ids), dim=1)

X_train, X_test, y_train, y_test = train_test_split(comb, labels, test_size=0.2, random_state=42)

# train_dataset = LMDataset(X_train, y_train)
# test_dataset = LMDataset(X_test, y_test)

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)
# test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

# # ids,attn_mask,ttype_ids,labels = next(iter(train_dataloader))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device == 'cuda':
#     torch.cuda.empty_cache()

# print("RUNNING ON:", device)

# # This is the LM that the sentiment was generated from
# model = AutoModelForSequenceClassification.from_pretrained("./finbertlm", num_labels=1)

# model = model.to(device=device)

# optimizer = AdamW(model.parameters(), lr=5e-5)
# ce_loss = nn.BCEWithLogitsLoss()

# sig = nn.Sigmoid()

# epochs = 3

# model.train()
# print("ENTERING TRAIN...")
# for epoch in range(epochs):
    
#     for ids,attn_mask,ttype_ids,labels in train_dataloader:
#         ids = ids.to(device=device)
#         attn_mask = attn_mask.to(device=device)
#         ttype_ids = ttype_ids.to(device=device)
#         labels = labels.to(device=device).unsqueeze(1)

#         outputs = model(input_ids=ids, attention_mask=attn_mask, token_type_ids=ttype_ids)
        
#         loss = ce_loss(outputs.logits, labels)
#         loss.backward()

#         optimizer.step()
#         optimizer.zero_grad()

#     if epoch % 3 == 0:
#         print(f"epoch: {epoch} loss: {loss}")
    
# model.eval()
# correct = 0
# total = 0

# print("Testing...")
# for ids,attn_mask,ttype_ids,labels in test_dataloader:
#     ids = ids.to(device=device)
#     attn_mask = attn_mask.to(device=device)
#     ttype_ids = ttype_ids.to(device=device)
#     labels = labels.to(device=device)

#     outputs = model(input_ids=ids, attention_mask=attn_mask, token_type_ids=ttype_ids)
#     class_pred = sig(outputs.logits)
    
#     pred = int(class_pred >= 0.5)
#     correct += sum(pred == labels)
#     total += len(labels)

# print(f"ACC: {correct/total}")
