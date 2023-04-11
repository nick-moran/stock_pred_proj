
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloaders.finbert_dataloaders import ValDataset
import torch.nn as nn

raw_news_df = pd.read_csv("./datasets/news_data_unique.csv")
raw_news_df.drop_duplicates(subset=['text'])

text = raw_news_df['text'].apply(lambda x: x[:600]).tolist()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# keys = ['input_ids', 'token_type_ids', 'attention_mask']
tokens_encoding = tokenizer(text[:2], padding='max_length', truncation=True, return_tensors="pt")

print(text[:2])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    torch.cuda.empty_cache()

print("RUNNING ON:", device)

tokens_encoding = tokens_encoding.to(device=device)

val_loader = ValDataset(tokens_encoding)
val_dataloader = DataLoader(val_loader, shuffle=False, batch_size=2)

# The model will give softmax outputs for three labels: positive, negative or neutral (in this order)
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model = model.to(device=device)
model.eval()

softmax = nn.Softmax(dim=1)

for ids,attn_mask,ttype_ids in val_dataloader:
    ids = ids.to(device=device)
    attn_mask = attn_mask.to(device=device)
    ttype_ids = ttype_ids.to(device=device)

    output = model(input_ids=ids, attention_mask=attn_mask, token_type_ids=ttype_ids)
    output = softmax(output.logits)
    print(output)
