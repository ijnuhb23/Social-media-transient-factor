# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

import warnings
warnings.filterwarnings("ignore")

# get the low and high score portfolios compositions
dataewma3low = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Low_quintile_portfolio//Low_quintile_portfolio.csv')
dataewma3high = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//High_quintile_portfolio//High_quintile_portfolio.csv')

# get the returns data
datafilt = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

# get market cap for each stock
market_capdata = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Market_Cap_Stocks//Market_Cap_Stocks.csv')


# list of stocks
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','VRSN','SWKS','DOCU','SPLK','OKTA','CEG'
] # put CEG at the end since it has no market cap data ## might not include anywhere because of this

## section to put returns on a monthly basis
# monthly dates of targeted timeframe
monthly_dates = [
    
"2015-01-01","2015-02-01","2015-03-01","2015-04-01","2015-05-01","2015-06-01","2015-07-01","2015-08-01","2015-09-01","2015-10-01","2015-11-01","2015-12-01",
"2016-01-01","2016-02-01","2016-03-01","2016-04-01","2016-05-01","2016-06-01","2016-07-01","2016-08-01","2016-09-01","2016-10-01","2016-11-01","2016-12-01",
"2017-01-01","2017-02-01","2017-03-01","2017-04-01","2017-05-01","2017-06-01","2017-07-01","2017-08-01","2017-09-01","2017-10-01","2017-11-01","2017-12-01",
"2018-01-01","2018-02-01","2018-03-01","2018-04-01","2018-05-01","2018-06-01","2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01","2018-12-01",
"2019-01-01","2019-02-01","2019-03-01","2019-04-01","2019-05-01","2019-06-01","2019-07-01","2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
"2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01","2020-06-01","2020-07-01","2020-08-01","2020-09-01","2020-10-01","2020-11-01","2020-12-01",
"2021-01-01","2021-02-01","2021-03-01","2021-04-01","2021-05-01","2021-06-01","2021-07-01","2021-08-01","2021-09-01","2021-10-01","2021-11-01","2021-12-01",
"2022-01-01","2022-02-01","2022-03-01","2022-04-01","2022-05-01","2022-06-01","2022-07-01"

]

# get market cap values for each stock
market_capdata = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Market_Cap_Stocks//Market_Cap_Stocks.csv')

#create new dataframe to store monthly scores
datafiltmonth = pd.DataFrame()

#create dataframe to perform interim manipulations
datafiltmanip =pd.DataFrame()

datafilt

# loop through monthly dates apart from last value to get monthly time series
for j in monthly_dates[:-1]:
    print(j)
    print(monthly_dates.index(j))

    # get the start date in the current loop
    start_date = monthly_dates[monthly_dates.index(j)]
    # get the end date in the current loop
    end_date = monthly_dates[monthly_dates.index(j)+1]

    # extract the monthly interval data
    mask = (datafilt['weekly date of Stock AAPL'] > start_date) & (datafilt['weekly date of Stock AAPL'] <= end_date)
    datafiltmanip = datafilt.loc[mask]

    # datafiltmanip

    # loop to calculate the monthly return for each stock 
    for i in stock_list:
        stock_name = str(i)

        # calculate the monthly return for each stock 
        datafiltmanip['monthly return of Stock '+str(stock_name)] = datafiltmanip['weekly return of Stock '+str(stock_name)].product()

    # only keep the first row of data, since they all contain the same values and makes it easier for appending with the other data points
    datafiltmanip = datafiltmanip.loc[datafiltmanip.index[0]]

    # identify each month with the starting date of the quarter
    datafiltmanip['monthly dates'] = start_date


    datafiltmonth = datafiltmonth.append(datafiltmanip)

datafiltmonth.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Monthly_Returns//All_Stocks_Monthly_Returns.csv')

## section to create high and low score portfolio returns

# create dataframes to include returns by copying the scores dataframe
dataewma3retlow = datafilt.copy()
dataewma3rethigh = datafilt.copy()

# replace na with to make loop easier
dataewma3high = dataewma3high.fillna(0)
dataewma3low = dataewma3low.fillna(0)

# get index list to be able to create the 1 month lag input
listindex = datafilt.index
listindex

# loop to extract relevant monthly stock returns for each portfolio 
for j in datafilt.index[:-1]:
    # print(j)


    for k in dataewma3high.index[:-1]:
        #listindex[k+1]
        #dataewma3high.iat[k,0]

        if dataewma3high.iat[k,0] == j:
            
            for i in stock_list:
                stock_name = str(i)
                

                dataewma3rethigh.loc[261,'weekly return of Stock '+str(stock_name)] = np.nan     
                # condition to remove returns values if not in the quintile high score portfolio 
                if dataewma3high.loc[k,'ewma of Stock '+str(stock_name)] == 0:
                    dataewma3rethigh.loc[listindex[k+1],'weekly return of Stock '+str(stock_name)] = np.nan

                dataewma3retlow.loc[261,'weekly return of Stock '+str(stock_name)] = np.nan
                # condition to remove returns values if not in the quintile low score portfolio 
                if dataewma3low.loc[k,'ewma of Stock '+str(stock_name)] == 0:
                    dataewma3retlow.loc[listindex[k+1],'weekly return of Stock '+str(stock_name)] = np.nan


# create two new dataframes to make data cleaner
dataewma3rethigh2 = pd.DataFrame()
dataewma3retlow2 = pd.DataFrame()

# loop to get all weekly returns
for i in stock_list:
    stock_name = str(i) 

    dataewma3rethigh2['weekly return of Stock '+str(stock_name)] = dataewma3rethigh['weekly return of Stock '+str(stock_name)]

    dataewma3retlow2['weekly return of Stock '+str(stock_name)] = dataewma3retlow['weekly return of Stock '+str(stock_name)]

dataewma3rethigh2['weekly dates'] = dataewma3rethigh['weekly date of Stock']
dataewma3retlow2['weekly dates'] = dataewma3retlow['weekly date of Stock']

# replace na with to make loop easier
dataewma3rethigh2 = dataewma3rethigh2.fillna(0)
dataewma3retlow2 = dataewma3retlow2.fillna(0)

# loop to put market cap values for stocks in both quintile portfolios
for j in dataewma3rethigh2.index:
   #  print(j)

    for i in stock_list:
        stock_name = str(i) 

        # condition for the high sentiment score portfolio
        if dataewma3rethigh2.loc[j,'weekly return of Stock '+str(stock_name)] != 0:
            dataewma3rethigh2.loc[j,'Market Cap of '+str(stock_name)] = market_capdata['Market Cap of '+str(stock_name)].values

        # condition for the low sentiment score portfolio
        if dataewma3retlow2.loc[j,'weekly return of Stock '+str(stock_name)] != 0:
            dataewma3retlow2.loc[j,'Market Cap of '+str(stock_name)] = market_capdata['Market Cap of '+str(stock_name)].values

# calculate the total weekly market cap for both portfolios
dataewma3rethigh2['Total Market Cap'] = dataewma3rethigh2.filter(regex='Market Cap').sum(axis=1)
dataewma3retlow2['Total Market Cap'] = dataewma3retlow2.filter(regex='Market Cap').sum(axis=1)

# calculate the percentage of the market cap for each stock for the high score portfolio
for j in dataewma3rethigh2.index:
    #print(j)
    # calculate the market cap percentage for every stock
    for i in dataewma3rethigh2:

        Stock_name = i.split()
        Stock_name = Stock_name[-1]

        
        if 'Market Cap of' in i:
            dataewma3rethigh2.loc[j,'Percentage of Market Cap for ' + Stock_name] = dataewma3rethigh2.loc[j,i]/dataewma3rethigh2.loc[j,'Total Market Cap']
            dataewma3rethigh2.loc[j,'Weight adjusted return of '+ Stock_name] = dataewma3rethigh2.loc[j,'Percentage of Market Cap for ' + Stock_name]*dataewma3rethigh2.loc[j,'weekly return of Stock ' + Stock_name]


# calculate the percentage of the market cap for each stock for the low score portfolio
for j in dataewma3retlow2.index:
    #print(j)
    for i in dataewma3retlow2:

        Stock_name = i.split()
        Stock_name = Stock_name[-1]
        
        if 'Market Cap of' in i:
            dataewma3retlow2.loc[j,'Percentage of Market Cap for ' + Stock_name] = dataewma3retlow2.loc[j,i]/dataewma3retlow2.loc[j,'Total Market Cap']
            dataewma3retlow2.loc[j,'Weight adjusted return of '+ Stock_name] = dataewma3retlow2.loc[j,'Percentage of Market Cap for ' + Stock_name]*dataewma3retlow2.loc[j,'weekly return of Stock ' + Stock_name]


## section to calculate the final weighted adjusted portfolio for low and high sentiment scores
for j in dataewma3rethigh2.index:
    #print(j)

    dataewma3rethigh2.loc[j,'Total weight adjusted portfolio'] = dataewma3rethigh2.loc[j].filter(regex='Weight adjusted return').sum(axis=0)

for j in dataewma3retlow2.index:
    #print(j)

    dataewma3retlow2.loc[j,'Total weight adjusted portfolio'] = dataewma3retlow2.loc[j].filter(regex='Weight adjusted return').sum(axis=0)


dataewma3rethigh2.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testing.csv')
dataewma3retlow2.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testingg.csv')

# create new dataframe for sentiment factor
dataewmasentfactor = pd.DataFrame()

# subtract high sentiment returns portfolio with the low returns portfolio
dataewmasentfactor['weekly sentiment factor'] = dataewma3rethigh2['Total weight adjusted portfolio']-dataewma3retlow2['Total weight adjusted portfolio']

dataewmasentfactor['weekly sentiment factor'] = dataewmasentfactor['weekly sentiment factor']*100


dataewmasentfactor['weekly dates'] = dataewma3retlow2['weekly dates']


dataewmasentfactor.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sentiment_factor//Sentiment_factor.csv')









