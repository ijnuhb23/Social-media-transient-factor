# imports needed Yahoo API extract

from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
from yahoo_fin.stock_info import get_data
from yahoo_fin.stock_info import get_quote_table

# list of stock to get historical price data
stock_list = ["GME","AMC","BB","NOK","BBBY"]

#Completeindex = get_data(str('AAPL'), start_date="01/04/2010", end_date="07/06/2022", index_as_date = True, interval="1wk")
#Completeindex = pd.DataFrame(index=Completeindex.index.copy())

# stock_list = ['AAPL']

# loop to extract historical data from Yahoo Finance
for i in stock_list:

    # set the stock name for each iteration
    stock_name = str(i)
    print(stock_name)

    # stock_weekly = pd.DataFrame(index=Completeindex.index.copy())

    # get all the historical stock price data
    stock_daily= get_data(str(stock_name), start_date="01/04/2010", end_date="07/06/2022", index_as_date = True, interval="1d")

    # get the market cap data for each stock to create the weights
    general_data = get_quote_table(str(stock_name))

    # create a new column for later
    stock_daily['daily return'] = stock_daily['adjclose']

    # create a loop to get weekly returns and add the results to the new column previously created
    for j in range(len(stock_daily)):
        
        # loop to get weekly return
        stock_daily['daily return'][j] = stock_daily['adjclose'][j]/stock_daily['adjclose'][j-1]

    # remove first value from return calculation list
    stock_daily['daily return'][0] = nan

    # add market cap to the file
    stock_daily['market cap'] = general_data['Market Cap']

    # add empty rows if date isn't equal to 2010-01-04
    # if stock_weekly['weekly return'].loc[0] == 
    
    # store the dataframe into a csv file
    stock_daily.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Topic_Yahoo_Finance_Historical_Prices//'+str(stock_name)+'_Topic_Yahoo_Finance_Historical_Prices.csv')
