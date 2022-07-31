# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

## section to put scores on a monthly basis
# read all filtered stock data
datanorm = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_summarised_normalised_VADER_scores//All_summarised_normalised_VADER_scores.csv')

datanorm

# read all filtered stock data
datafiltret = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

# transpose weekly dates to the datanorm dataframe
datanorm['weekly dates'] = datafiltret['weekly date of Stock']

#create new dataframe to store monthly scores
datanormmonth = pd.DataFrame()

#create dataframe to perform interim manipulations
datanormmanip =pd.DataFrame()

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

# stock list used for list
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]

# loop through monthly dates apart from last value to get monthly time series
for j in monthly_dates[:-1]:
    print(j)
    print(monthly_dates.index(j))

    # get the start date in the current loop
    start_date = monthly_dates[monthly_dates.index(j)]
    # get the end date in the current loop
    end_date = monthly_dates[monthly_dates.index(j)+1]

    # extract the monthly interval data
    mask = (datanorm['weekly dates'] > start_date) & (datanorm['weekly dates'] <= end_date)
    datanormmanip = datanorm.loc[mask]

    # loop each column to average the weekly scores
    for i in stock_list:
        stock_name = str(i)

        # calculate the mean of average weekly scores for each sector 
        datanormmanip['mean of Stock '+str(stock_name)] = datanormmanip['mean of Stock '+str(stock_name)].mean()

    # only keep the first row of data, since they all contain the same values and makes it easier for appending with the other data points
    datanormmanip = datanormmanip.loc[datanormmanip.index[0]]

    # identify each quarter with the starting date of the quarter
    datanormmanip['monthly dates'] = start_date


    datanormmonth = datanormmonth.append(datanormmanip)

datanormmonth.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Monthly_Scores//All_Stocks_Monthly_Scores.csv')

# create a dataframe that will contain the 3-year rolling exponential moving average (EWMA) sentiment scores
dataewma3 = pd.DataFrame()

# loop to obtain 3-year rolling EWMA
for i in stock_list:
        stock_name = str(i)

        dataewma3['ewma of Stock '+str(stock_name)] = datanorm['mean of Stock '+str(stock_name)].ewm(span=52,min_periods=35).mean()

dataewma3['weekly dates'] = datanorm['weekly dates']

## section to build quintile portfolios

# create dataframe for low scores quintile
dataewma3low = dataewma3.copy()

# create dataframe for high scores quintile
dataewma3high = dataewma3.copy()

# rank stocks by scores at each timeframe
dataewma3ranked = dataewma3.rank(axis=1,method='dense')

# find max value by row
dataewma3ranked['number of active stocks in week'] = dataewma3ranked.max(axis=1)

# find value to split into quintiles
dataewma3ranked['number of stocks per quintile'] = dataewma3ranked['number of active stocks in week']/5

dataewma3ranked['weekly dates'] = dataewma3['weekly dates']

dataewma3ranked.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testinggg.csv')

# loop to extract lowest and highest quintile values
for j in dataewma3ranked.index:
    for i in stock_list:
            stock_name = str(i)

            #dataewma3ranked.loc[j]['ewma of Stock '+str(stock_name)]
            #dataewma3.loc[j]['ewma of Stock '+str(stock_name)]
            #dataewma3ranked.loc[j]['ewma of Stock '+str(stock_name)] > dataewma3ranked.loc[j]['number of stocks per quintile']

            # condition to keep lowest quintile values
            if dataewma3ranked.loc[j]['ewma of Stock '+str(stock_name)] > dataewma3ranked.loc[j]['number of stocks per quintile']:
                dataewma3low.loc[j,'ewma of Stock '+str(stock_name)] = np.nan


            # condition to keep highest quintile values
            if dataewma3ranked.loc[j]['ewma of Stock '+str(stock_name)] < dataewma3ranked.loc[j]['number of active stocks in week']-dataewma3ranked.loc[j]['number of stocks per quintile']:
                dataewma3high.loc[j,'ewma of Stock '+str(stock_name)] = np.nan

# save both portfolio compositions
dataewma3low.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Low_quintile_portfolio//Low_quintile_portfolio.csv')
dataewma3high.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//High_quintile_portfolio//High_quintile_portfolio.csv')


