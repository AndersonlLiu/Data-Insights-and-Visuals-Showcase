#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 10 year treasuty rate

from pathlib import Path
import sys
import pandas as pd
import requests

def getTreasuryPrice():
    response_treasury = requests.get('https://data.nasdaq.com/api/v3/datasets/USTREASURY/YIELD.json?api_key=m4QDcyVZJxsukKV23FsE')

    t_js = response_treasury.json()
    t_data = t_js.get('dataset')['data']
    t_price = pd.DataFrame(columns = ['date', 'close_price'])
    for i in t_data:
        length = len(t_price)
        t_price.loc[length] = [i[0], i[10]]
    
    t_price = t_price[t_price['date'] > "2020-02-20"]
    t_price['pct_change'] = t_price['close_price'].pct_change()
    t_price = t_price.drop(0)
    return t_price

data = getTreasuryPrice()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("print the whole dataset")
        print(data)

    elif len(sys.argv) == 2 and sys.argv[1] == '--scrape':
        print("print the first 5 rows")
        print(data[:5])
    
    elif sys.argv[1] == '--static':
        print("download the dataset into csv file")
        filepath = Path('data/Treasury_Price.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        data.to_csv(filepath, index = False) 

else: # Print a statement for the wrong/unrecognized arguments input
	print("The arguments do not match the requirements. Please refer README.md for understanding the requirements")

