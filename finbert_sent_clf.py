from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloaders.finbert_dataloaders import ValDataset
import torch.nn as nn
from datetime import datetime, timedelta
import calendar
import numpy as np
from tqdm.auto import tqdm

stock_time_format = '%H:%M'
date_format = '%Y-%m-%d'
news_time_format = '%H:%M:%S'
dtime_gen = lambda x,y: datetime.combine(x,y.time())

news_ds = pd.read_csv("./datasets/news_data_unique.csv")

stocks_ds = pd.read_csv("./charts/APPLE60.csv")
stocks_ds['dir'] = np.sign(stocks_ds['close'] - stocks_ds['open'])
stocks_ds['time'] = stocks_ds['time'].apply(lambda x:  datetime.strptime(x, stock_time_format))
stocks_ds['date'] = stocks_ds['date'].apply(lambda x:  datetime.strptime(x, date_format))
stocks_ds['dtime'] = stocks_ds.apply(lambda x: dtime_gen(x.date, x.time), axis=1)
news_ds['time'] = news_ds['time'].apply(lambda x:  datetime.strptime(x, news_time_format))
news_ds['date'] = news_ds['date'].apply(lambda x:  datetime.strptime(x, date_format))
news_ds['dtime'] = news_ds.apply(lambda x: dtime_gen(x.date, x.time), axis=1)
stocks_ds = stocks_ds[stocks_ds['date'] >= datetime.strptime("2018-01-01", date_format)]

pred = []
answer = []

# finbert
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
if device == 'cuda':
    torch.cuda.empty_cache()

print("RUNNING ON:", device)

# The model will give softmax outputs for three labels: positive, negative or neutral (in this order)
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model = model.to(device=device)
model.eval()
softmax = nn.Softmax(dim=1)

progress_bar = tqdm(range(len(stocks_ds)))

for row in stocks_ds.itertuples():
    date = row.date
    time = row.time
    dtime = row.dtime
    
    day_of_week = calendar.day_name[date.weekday()]

    if time > datetime.strptime('20:30', stock_time_format) or time < datetime.strptime('13:30', stock_time_format):
       continue

    if day_of_week == "Monday" and time == datetime.strptime('13:30', stock_time_format):
        t_w_l = dtime - timedelta(hours=3+48+13, minutes=0)
    
    elif time == datetime.strptime('13:30', stock_time_format):
      t_w_l = dtime - timedelta(hours=16, minutes=0)

    else:
       t_w_l = dtime - timedelta(hours=1, minutes=0)

    mask = (news_ds['dtime'] > t_w_l) & (news_ds['dtime']  < dtime)
    news_for_hour = news_ds[mask]['text'].tolist()

    try:
      news_for_hour = news_for_hour[:8]
      tokens_encoding = tokenizer(news_for_hour, max_length=48, padding='max_length', truncation=True, return_tensors="pt")
      tokens_encoding = tokens_encoding.to(device=device)

      outputs = model(**tokens_encoding)
      outputs = softmax(outputs.logits)
      output_idx = outputs.argmax(dim=1)
      output_idx[output_idx == 1] = -1
      output_idx[output_idx == 0] = 1
      output_idx[output_idx == 2] = 0
      sent_for_hour = np.sign(output_idx.sum())

      pred.append(sent_for_hour)
      answer.append(row.dir)
    except Exception as e:
       pass


    progress_bar.update(1)

pred = np.array(pred)
answer = np.array(answer)

correct = sum(pred == answer)
print(f"ACC: {correct/len(answer)}")