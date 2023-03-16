import pandas as pd
import datetime
import calendar

def get_hour_news(date, hours, news_data, mode='open'):
    #get news from this day, after market close
    news_day = news_data[news_data['date'] == date]
    
    #a good idea might be to get the indicies rather than the actual full frames
    #open market hours
    if mode == 'open':
      news_by_hour = {i: news_day[news_day['time'].dt.hour == i ] for i in hours}
    else:
        news_by_hour = [news_day[news_day['time'].dt.hour == i ] for i in hours]

    return news_by_hour

def get_hourly_news_by_date(date, stock_chart, news_data):
    """
    input: date string in the form YYYY-MM-DD

    this function returns the hourly news for a given date.
    if the date is a weekend, it returns nothing.

    if an article comes out after hours or on a weekend, it gets attached to
    the next opening day's articles.

    note that market hours are listed at 8am to 3pm. this is so that we can shift the
    hours up by one to be used as described in the prof's paper.
    """

    open_dates = list(stock_chart['date'].apply(lambda x: x.replace('.', '-')))

    pre_market = [i for i in range(0,8)]
    market = [i for i in range(8,16)]
    post_market = [i for i in range(16,24)]

    market_times = [pre_market, market, post_market]

    
    if date in open_dates:
        print('stock market was open on:', date)
    else:
        print('stock market was closed on:', date)
        return []
    
    day_of_week = calendar.day_name[datetime.datetime.strptime(date, '%Y-%m-%d').weekday()]
    
    #if monday, we'll have to get all the after hours news from the past two days as well.
    if day_of_week == "Monday":
        today = open_dates.index(date)
        
        y_news = []
        for i in range(1,4):
            past_day = open_dates[today-i]
            if i != 3:
                for j in range(3):
                    y_news.extend(get_hour_news(past_day,market_times[j],news_data,'wknd'))
            else:
                y_news.extend(get_hour_news(past_day,market_times[2],news_data,'wknd'))

    else:
        #get yesterday's date
        today = open_dates.index(date)
        yesterday = open_dates[today-1]

        y_news = get_hour_news(yesterday,post_market,news_data,'yest')
    
    
    p_news = get_hour_news(date,pre_market,news_data,'pre')

    early_news = y_news + p_news
    early_news = pd.concat(early_news)
    
    #get today's hours
    market_news = get_hour_news(date,market,news_data,'open')
    
    #add premarket
    market_news[8] = pd.concat([early_news, market_news[8]])
    

    return market_news
    


stock_chart = pd.read_csv('../charts/APPLE1440.csv')
news_data = pd.read_csv('../datasets/news_data_filtered.csv')
news_data['time'] = pd.to_datetime(news_data['time'])


get_hourly_news_by_date('2018-05-07', stock_chart, news_data)
