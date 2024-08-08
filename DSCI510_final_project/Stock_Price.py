#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Stock historical price

# Get the data from API
import requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys

def getStockPrice():
    response = requests.get('https://api.stockdata.org/v1/data/eod?symbols=AAPL&api_token=jd0qoX9VUT5lMG5onfd3RBXzWEhXVKPwNz1Rheao')

    # Build the dataframe
    js = response.json()
    date = js.get('data')
    stock_price = pd.DataFrame(columns = ['date', 'close_price'])
    for i in date:
       stock_price_length = len(stock_price)
       stock_price.loc[stock_price_length] = [i['date'], i['close']]

    # Data cleaning
    stock_price['date'] = stock_price['date'].str.replace('T16:00:00.000000Z', '')
    stock_price = stock_price.sort_values(by=['date']).reset_index()
    stock_price = stock_price.drop(['index'], axis = 1)
    stock_price['pct_change'] = stock_price['close_price'].pct_change()
    stock_price = stock_price.drop(0)
    return stock_price

data = getStockPrice()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("print the whole dataset")
        print(data)

    elif len(sys.argv) == 2 and sys.argv[1] == '--scrape':
        print("print the first 5 rows")
        print(data[:5])
    
    elif sys.argv[1] == '--static':
        print("download the dataset into csv file")
        filepath = Path('data/Stock_Price.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        data.to_csv(filepath, index = False)

else: # Print a statement for the wrong/unrecognized arguments input
	print("The arguments do not match the requirements. Please refer README.md for understanding the requirements")
