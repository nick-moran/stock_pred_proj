
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloaders.finbert_dataloaders import ValDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split

news_df = pd.read_csv("./finbert_datasets/hourly_apple.csv")
# news_df.drop_duplicates(subset=['text'])
labels = news_df['dir']

text = news_df['text'].apply(lambda x: x[:600]).tolist()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# keys = ['input_ids', 'token_type_ids', 'attention_mask']
tokens_encoding = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
ids = tokens_encoding['input_ids']
attn_mask = tokens_encoding['attention_mask']
ttype_ids = tokens_encoding['token_type_ids']

comb = torch.stack((ids,attn_mask,ttype_ids), dim=1)

X_train, X_test, y_train, y_test = train_test_split(comb, labels, test_size=0.2, random_state=42)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device == 'cuda':
#     torch.cuda.empty_cache()

# print("RUNNING ON:", device)


# val_loader = ValDataset(tokens_encoding)
# val_dataloader = DataLoader(val_loader, shuffle=False, batch_size=2)

# # This is the LM that the sentiment was generated from
# model = AutoModelForSequenceClassification.from_pretrained("./finbertlm", num_labels=2)
# model = model.to(device=device)
# model.eval()

# softmax = nn.Softmax(dim=1)

# for ids,attn_mask,ttype_ids in val_dataloader:
#     ids = ids.to(device=device)
#     attn_mask = attn_mask.to(device=device)
#     ttype_ids = ttype_ids.to(device=device)

#     output = model(input_ids=ids, attention_mask=attn_mask, token_type_ids=ttype_ids)
#     output = softmax(output.logits)
#     print(output)
