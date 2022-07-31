# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

# read all filtered stock data
datafilt = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_summarised_VADER_scores//All_summarised_Topic_VADER_scores.csv')

datafilt

# stock list used for list
stock_list = ["GME","AMC","BB","NOK","BBBY"]

# loop to store summary data from each stock into the main dataframe
for i in stock_list:
    stock_name = str(i)
    # print(stock_name)

    # normalise and standardise the mean scores columns of each stock sector with a rolling window of 1 year on a weekly basis
    window = 52
    min_periods = 5
    target_column = stock_name
    roll = datafilt['mean of Stock '+str(stock_name)].rolling(window, min_periods)
    datafilt['mean of Stock '+str(stock_name)] = (datafilt['mean of Stock '+str(stock_name)] - roll.mean()) / roll.std()

datafilt.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_summarised_normalised_VADER_scores//All_summarised_Topic_normalised_VADER_scores.csv')