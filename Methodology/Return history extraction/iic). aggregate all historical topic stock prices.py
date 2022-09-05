# imports for aggregation

from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt

# import list of active stocks in portfolio
stock_list = ["GME","AMC","BB","NOK","BBBY"]

# create dataframe to contain market caps of all stocks (as of June 2022)
datamarketcap = pd.DataFrame()

# create dataframe to contain all return history
datareturntot = pd.DataFrame()

# add all the possible dates for this dataframe
data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Topic_Yahoo_Finance_Historical_Prices//BBBY_Topic_Yahoo_Finance_Historical_Prices.csv')
datareturntot['daily date of Stock'] = data['Unnamed: 0']

# loop to store summary data from each stock into the main dataframe
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # open each file containing the summarised scores 
    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Topic_Yahoo_Finance_Historical_Prices//'+str(stock_name)+'_Topic_Yahoo_Finance_Historical_Prices.csv')

    # add the weekly dates of each stock
    datareturntot['daily date of Stock '+str(stock_name)] = np.nan
    
    # add the weekly returns of each stock
    datareturntot['daily return of Stock '+str(stock_name)] = np.nan

    # loop to match the dates of daily returns when the stock didn't exist at the start date of 2010
    if datareturntot['daily date of Stock'].loc[0] != data['Unnamed: 0'].loc[0]:
        for j in datareturntot.index:
            for k in data.index:
                if datareturntot['daily date of Stock'].loc[j] == data['Unnamed: 0'].loc[k]:
                    datareturntot['daily date of Stock '+str(stock_name)].loc[j] = data['Unnamed: 0'].loc[k]
                    datareturntot['daily return of Stock '+str(stock_name)].loc[j] = data['daily return'].loc[k]
    else:
        datareturntot['daily date of Stock '+str(stock_name)] = data['Unnamed: 0']
        datareturntot['daily return of Stock '+str(stock_name)] = data['daily return']



    # add the weekly dates of each stock
    # datareturntot['weekly date of Stock '+str(stock_name)] = np.nan

    # add the mean scores columns of each stock
    # ['weekly return of Stock '+str(stock_name)] = data['weekly return']



datareturntot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_Yahoo_Historical_prices//All_stock_Topic_Yahoo_Historical_prices.csv')

#datareturntot['weekly return of Stock ABNB']