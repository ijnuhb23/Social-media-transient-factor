# imports needed for filter


from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt

# filter data comes from: https://siblisresearch.com/data/historical-components-nasdaq/
# read filter data which indicates which stocks are in the NASDAQ 100
filter = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Nasdaq_components_historical//Nasdaq_components_historical.csv')

# read dataset consider all summarised scores of each stock
datatot= pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_prices//All_stock_Yahoo_Historical_prices.csv')

# first time frame
first_date = int(dt.datetime(2010,1,4,0,0).timestamp())

# create list for epoch time frames
weeklytimeframes_list = []
list = []

# add all the weekly epoch time frames to the list and transform into normal dates
for i in range(750):
    date = first_date+i*604800
    date = dt.datetime.fromtimestamp(date)
    weeklytimeframes_list.append(date)
    list.append(i)

datatot
datatot['weekly date of Stock'][259]


# merge both to get dates for each row number
weeklytimeframes_list_merged = pd.DataFrame()
weeklytimeframes_list_merged = weeklytimeframes_list_merged.append(weeklytimeframes_list)

stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]

# AZN was added recently in FEB 2022, was added to this year
# ODFL was added recently in JAN 2022, was added to this year
# CEG was added recently in FEB 2022, was added to this year

for j in stock_list:
    stock_name = str(j)
    print(stock_name)

    row = []

    row = filter.loc[filter['Ticker'] == stock_name]

    row
    
    if row['12/31/2014'].item() != 'X':

        for i in range(259,311,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2015'].item() != 'X':

        for i in range(311,363,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2016'].item() != 'X':

        for i in range(363,415,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2017'].item() != 'X':

        for i in range(415,467,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2018'].item() != 'X':

        for i in range(467,520,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2019'].item() != 'X':

        for i in range(520,572,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2020'].item() != 'X':

        for i in range(572,624,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan

    if row['12/31/2021'].item() != 'X':

        for i in range(624,649,1):
            datatot['weekly date of Stock '+str(stock_name)][i] = np.nan
            datatot['weekly return of Stock '+str(stock_name)][i] = np.nan


datatot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')
