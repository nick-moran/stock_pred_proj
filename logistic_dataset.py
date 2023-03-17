import numpy as np
import pandas as pd
from utils.news_utils import get_hourly_news_by_date

def create_dataset(ticker, news_data):
    """
    Given a ticker and a filtered news corpus, this outputs two datasets:
      - the tf-idf matrix for each example
      - the chart labels
    
    the data in the dataset is hourly stock data and 
    the first day in the dataset is 2018-08-02
    """

    hourly_stocks = pd.read_csv(f'./charts/{ticker}60.csv')
    hourly_stocks = hourly_stocks[hourly_stocks['date'] >= '2018-09-31']

    hourly_stocks['dir'] = np.sign(hourly_stocks['close'] - hourly_stocks['open'])
    
   
    #use this just to get days stock market is open
    stock_days = pd.read_csv(f'./charts/{ticker}1440.csv')
    stock_days = stock_days[stock_days['date'] >= '2018-09-31']
    
    dataset = []
    for day in stock_days['date'].tolist():
        # print(day)
        hourly_stocks_day = hourly_stocks[hourly_stocks['date'] == day]
        
        #sometimes there's less hours apparently 
        if len(hourly_stocks_day) > 8:
            hourly_stocks_day = hourly_stocks_day[:8]


        news = get_hourly_news_by_date(day, stock_days, news_data)


        for i in range(len(hourly_stocks_day)):
            news_hour = i+8
            news[news_hour]['dir'] = [hourly_stocks_day.iloc[i]['dir']]*len(news[news_hour])
            dataset.extend([news[news_hour][['text','dir']]])

    
    dataset = pd.concat(dataset)

    dataset.to_csv(f'./log_datasets/{ticker}_text_hourly.csv',index=False)

    
news_data = pd.read_csv('./datasets/news_data_filtered.csv')
news_data['time'] = pd.to_datetime(news_data['time'])

create_dataset('APPLE',news_data)