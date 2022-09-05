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
datafilt = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_summarised_filter_VADER_scores//All_summarised_filtered_VADER_scores.csv')

datafilt

# stock list used for list
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]


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

datafilt.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_summarised_normalised_VADER_scores//All_summarised_normalised_VADER_scores.csv')