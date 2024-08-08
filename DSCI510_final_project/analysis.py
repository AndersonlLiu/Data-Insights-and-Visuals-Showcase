from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from math import sin
from math import radians
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.metrics import accuracy_score
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from math import sin
from math import radians
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from statistics import mean
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_stock_price():
    data = pd.read_csv('data/Stock_Price.csv')
    return data
def get_news():
    return pd.read_csv('data/News.csv')
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

def analysis():
    stock_price = get_stock_price()
    # NA value test
    stock_price.isnull().sum()

    # Plot the Stock Price vs. date
    plt.plot(stock_price['date'], stock_price['close_price'], color='red')
    plt.title('AAPL Stock Price Vs Date', fontsize=14)
    plt.ylabel('AAPL Stock Price', fontsize=14)
    plt.figure(figsize=(15, 3))
    plt.show()

    # Change Date to the index
    stock_price.index = stock_price.date
    data_SERIES = stock_price['pct_change']

    # Stationary Test for the percent change
    data_new = stock_price[['date', 'pct_change']].set_index(['date'])
    def test_stationarity(timeseries):
        # ADF Test (stationarity test)
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used',
                                                'Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    test_stationarity(data_new)

    # If the p-value is not significant(<<5), than it is stationary

    # White Noise Test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print('White Noise for pct change：',acorr_ljungbox(data_SERIES,lags=1))

    # If the P value is larger than 0.05, than this time seires is a white noise data, which means we should discontinue the analysis
    # and change our data

    # Change the time seiers data from percent change to the stock price

    # Stationary Test for the stock price
    data_closePrice = stock_price[['date', 'close_price']].set_index(['date'])
    test_stationarity(data_closePrice)
    # The p value is too large so is not stationary, so we have to make difference of the data to make it stationary
    # to conduct useful analysis

    close = data_closePrice['close_price'].tolist()
    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return diff
    # difference the dataset
    diff = difference(close, 20)

    # ADF Test (stationarity test)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(diff, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used',
                                            'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    from statsmodels.stats.diagnostic import acorr_ljungbox
    print('White Noise for difference：',acorr_ljungbox(diff,lags=1))

    # The Result shows that this is a stationary time series data and is not white noise, so we can conduct further analysis:)

    # Preparing data for time series
    date = stock_price['date'].tolist()[20:]
    d = {'date': date,'price_diff': diff}
    df = pd.DataFrame(d)
    df = df.set_index(['date'])

    # Plot the ACF and PACF graph to determine which model to use (AR / MA / ARIMA)
    plot_acf(df)
    plot_pacf(df)
    plt.show()

    # Due the ACF graph decrease to 0 quickly and PACF become 0 suddenly after the second data, we use AR model to forecast

    from statsmodels.tsa.ar_model import AutoReg
    ar_model = AutoReg(df, lags=2).fit()
    pred = ar_model.predict(start=len(df), end=(len(df) + 30))

    resid = ar_model.resid
    print(f'dw index is: {sm.stats.durbin_watson(resid.values)}')
    # the model is better when the result is close to 2

    # Draw the predction
    plt.figure(figsize=(24, 8))   
    orig = plt.plot(df['price_diff'], color='blue',label='Original')
    predict = plt.plot(pred, color='red',label='Predict')
    plt.legend(loc='best')
    plt.title('Original&Predict')
    plt.show(block=False)

    # Sentiment Analysis
    news = get_news()
    analyzer = SentimentIntensityAnalyzer()
    scores = news['News Headlines'].apply(analyzer.polarity_scores).tolist()
    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')
    score = news['compound'].tolist()

    # Plot the sentiment anlysis score and return the mean of it
    plt.figure(figsize=(24, 8)) 
    plt.scatter(range(0,100), score)
    plt.ylabel('sentiment analysis score')
    plt.title('Sentiment Analysis Of Stock Market ')
    plt.grid(True)
    plt.show()
    print(f'the mean of the sentiment score is {mean(score)}')

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


if __name__ == '__main__':
    analysis()
    analysis_excess()