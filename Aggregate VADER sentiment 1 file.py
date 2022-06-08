# imports needed for VADER


from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt

# import list of active stocks in portfolio
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]

#create empty dataframe
datatot = pd.DataFrame()

datatot

# loop to store summary data from each stock into the main dataframe
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_summarised_VADER_scores//'+str(stock_name)+'_summarised_VADER_scores.csv')

    data.columns = ['','mean of Stock '+str(stock_name),'count of Stock '+str(stock_name) ]

    datatot['mean of Stock '+str(stock_name)] = data['mean of Stock '+str(stock_name)]

    datatot['count of Stock '+str(stock_name)] = data['count of Stock '+str(stock_name)]

    datatot


datatot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_summarised_VADER_scores//All_summarised_VADER_scores.csv')
