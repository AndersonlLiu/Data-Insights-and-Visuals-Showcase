#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# News

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path 

def getnews():
    finviz_url = 'https://finviz.com/quote.ashx?t=AAPL'
    news_tables = {}
    req = Request(url=finviz_url,headers={'user-agent': 'my-app/0.0.1'}) 
    resp = urlopen(req)    
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables['AAPL'] = news_table
    
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text() 
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
               time = date_scrape[0]
            
            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]
        
            parsed_news.append([ticker, date, time, text])
        
    news = pd.DataFrame(parsed_news, columns = ['Stock', 'Date', 'Time', 'News Headlines'])
    news['Date'] = pd.to_datetime(news['Date'], format='%b-%d-%y')
    return news

data = getnews()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("print the whole dataset")
        print(data)

    elif len(sys.argv) == 2 and sys.argv[1] == '--scrape':
        print("print the first 5 rows")
        print(data[:5])
    
    elif sys.argv[1] == '--static':
        print("download the dataset into csv file")
        filepath = Path('data/News.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        data.to_csv(filepath, index = False) 

else: # Print a statement for the wrong/unrecognized arguments input
	print("The arguments do not match the requirements. Please refer README.md for understanding the requirements")


