import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

stock_time_format = '%H:%M'
date_format = '%Y-%m-%d'

news_time_format = '%H:%M:%S'

dtime_gen = lambda x,y: datetime.combine(x,y.time())

news_ds = pd.read_csv("./datasets/news_data_unique.csv")

print('news: ', len(news_ds))

stocks_ds = pd.read_csv("./charts/APPLE60.csv")
stocks_ds['dir'] = np.sign(stocks_ds['close'] - stocks_ds['open'])


stocks_ds['time'] = stocks_ds['time'].apply(lambda x:  datetime.strptime(x, stock_time_format))
stocks_ds['date'] = stocks_ds['date'].apply(lambda x:  datetime.strptime(x, date_format))

stocks_ds['dtime'] = stocks_ds.apply(lambda x: dtime_gen(x.date, x.time), axis=1)

news_ds['time'] = news_ds['time'].apply(lambda x:  datetime.strptime(x, news_time_format))
news_ds['date'] = news_ds['date'].apply(lambda x:  datetime.strptime(x, date_format))

news_ds['dtime'] = news_ds.apply(lambda x: dtime_gen(x.date, x.time), axis=1)


stocks_ds = stocks_ds[stocks_ds['date'] >= datetime.strptime("2018-01-01", date_format)]


dataset = []

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

    time_for_hour = news_ds[mask]['dtime'].tolist()

    dirs = [row.dir]*len(news_for_hour)


    data = np.c_[news_for_hour, dirs, time_for_hour]
    

    dataset.extend(data)


df = pd.DataFrame(dataset, columns=['text','dir', 'dtime'])


df.to_csv('./finbert_datasets/hourly_apple.csv')
print(df)


