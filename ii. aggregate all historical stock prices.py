# imports for aggregation

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

# create dataframe to contain market caps of all stocks (as of June 2022)
datamarketcap = pd.DataFrame()

# create dataframe to contain all return history
datareturntot = pd.DataFrame()

# add all the possible dates for this dataframe
data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Yahoo_Finance_Historical_Prices//AAPL_Yahoo_Finance_Historical_Prices.csv')
datareturntot['weekly date of Stock'] = data['Unnamed: 0']

# loop to store summary data from each stock into the main dataframe
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # open each file containing the summarised scores 
    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Yahoo_Finance_Historical_Prices//'+str(stock_name)+'_Yahoo_Finance_Historical_Prices.csv')

    # add the weekly dates of each stock
    datareturntot['weekly date of Stock '+str(stock_name)] = np.nan
    
    # add the weekly returns of each stock
    datareturntot['weekly return of Stock '+str(stock_name)] = np.nan

    if datareturntot['weekly date of Stock'].loc[0] != data['Unnamed: 0'].loc[0]:
        for j in datareturntot.index:
            for k in data.index:
                if datareturntot['weekly date of Stock'].loc[j] == data['Unnamed: 0'].loc[k]:
                    datareturntot['weekly date of Stock '+str(stock_name)].loc[j] = data['Unnamed: 0'].loc[k]
                    datareturntot['weekly return of Stock '+str(stock_name)].loc[j] = data['weekly return'].loc[k]
    else:
        datareturntot['weekly date of Stock '+str(stock_name)] = data['Unnamed: 0']
        datareturntot['weekly return of Stock '+str(stock_name)] = data['weekly return']



    # add the weekly dates of each stock
    # datareturntot['weekly date of Stock '+str(stock_name)] = np.nan

    # add the mean scores columns of each stock
    # ['weekly return of Stock '+str(stock_name)] = data['weekly return']



datareturntot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_prices//All_stock_Yahoo_Historical_prices.csv')

#datareturntot['weekly return of Stock ABNB']