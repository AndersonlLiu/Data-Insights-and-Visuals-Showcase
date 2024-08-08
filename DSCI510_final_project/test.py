import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean

def get_t_price():
    t_price = pd.read_csv('data/Treasury_Price.csv')
    s_price = pd.read_csv('data/Stock_Price.csv')
    s_price = s_price.set_index(['date'])
    t_price = t_price.set_index(['date'])
    merged_df = pd.merge(s_price, t_price, left_index=True, right_index=True)

     # Preparing the data
    merged_df.rename(columns = {'pct_change_x':'pct_change_stock'}, inplace = True)
    merged_df.rename(columns = {'pct_change_y':'pct_change_treasury'}, inplace = True)
    merged_df = merged_df.drop(merged_df.columns[[0, 2]], axis=1)
    merged_df['excess_return'] = merged_df['pct_change_stock'] - merged_df['pct_change_treasury']
    return merged_df

def analysis_excess():
    merged_df = get_t_price()
    mean_value = mean(merged_df['excess_return'].tolist())
    # Plot the excess return of the stock AAPL and calculat the mean of the excess return
    plt.figure(figsize=(24, 8)) 
    plt.ylabel('excess return rate of AAPL')
    plt.title('Excess Return')
    plt.plot(range(0,183), merged_df['excess_return'])
    plt.show()
    print(f'the average of the excess return rate is {mean_value}')

data = analysis_excess()

if __name__ == '__main__':
   analysis_excess()