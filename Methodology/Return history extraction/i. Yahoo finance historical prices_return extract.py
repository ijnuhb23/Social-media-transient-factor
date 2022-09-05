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
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]

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
    stock_weekly= get_data(str(stock_name), start_date="01/04/2010", end_date="07/06/2022", index_as_date = True, interval="1wk")

    # get the market cap data for each stock to create the weights
    general_data = get_quote_table(str(stock_name))

    # create a new column for later
    stock_weekly['weekly return'] = stock_weekly['adjclose']

    # create a loop to get weekly returns and add the results to the new column previously created
    for j in range(len(stock_weekly)):
        
        # loop to get weekly return
        stock_weekly['weekly return'][j] = stock_weekly['adjclose'][j]/stock_weekly['adjclose'][j-1]

    # remove first value from return calculation list
    stock_weekly['weekly return'][0] = nan

    # add market cap to the file
    stock_weekly['market cap'] = general_data['Market Cap']

    # add empty rows if date isn't equal to 2010-01-04
    # if stock_weekly['weekly return'].loc[0] == 
    
    # store the dataframe into a csv file
    stock_weekly.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Yahoo_Finance_Historical_Prices//'+str(stock_name)+'_Yahoo_Finance_Historical_Prices.csv')
