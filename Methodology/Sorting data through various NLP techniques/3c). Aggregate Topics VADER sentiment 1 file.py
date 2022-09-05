# imports needed for VADER


from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt

# import list of active stocks in portfolio
stock_list = ["GME","AMC","BB","NOK","BBBY"]

#create empty dataframe
datatot = pd.DataFrame()

# loop to store summary data from each stock into the main dataframe
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # open each file containing the summarised scores 
    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Topic_summarised_VADER_scores//'+str(stock_name)+'_summarised_VADER_scores.csv')

    # add the columns to input the new columnds from each stock
    data.columns = ['','mean of Stock '+str(stock_name),'count of Stock '+str(stock_name) ]

    # add the mean scores columns of each stock
    datatot['mean of Stock '+str(stock_name)] = data['mean of Stock '+str(stock_name)]

    # add the count columns of each stock
    datatot['count of Stock '+str(stock_name)] = data['count of Stock '+str(stock_name)].replace(0,np.nan)

    # datatot

# save the dataframe into a csv file
datatot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_summarised_VADER_scores//All_summarised_Topic_VADER_scores.csv')

