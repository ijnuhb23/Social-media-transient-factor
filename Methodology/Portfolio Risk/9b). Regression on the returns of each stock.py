# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp
import statsmodels.api as sm

# get relevant dataframes
dataewmasentfactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sentiment_factor//Sentiment_factor.csv')

datafamafactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Fama_French_Factors//F-F_Research_Data_5_Factors_2x3.csv')

datafiltweek = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')


# list of stocks
stock_list = ['AAPL','MSFT' ,'AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','VRSN','SWKS','DOCU','SPLK','OKTA','CEG']

# merge both dataframes together

dataallfactors = datafamafactor.copy()

dataallfactors['Sentiment factor'] = dataewmasentfactor['weekly sentiment factor']

datafiltweek = datafiltweek[:-3]
dataallfactors = dataallfactors[:-3]

# remove rows where sentiment factor doesn't exist yet
for i in dataallfactors.index:

    if dataallfactors.loc[i,'Sentiment factor'] == 0:

        dataallfactors = dataallfactors.drop(i)
        datafiltweek = datafiltweek.drop(i)
        continue
        
    if dataallfactors.loc[i,'Sentiment factor'] >= 90:

        dataallfactors = dataallfactors.drop(i)
        datafiltweek = datafiltweek.drop(i)
        continue

    if dataallfactors.loc[i,'Sentiment factor'] <= -90:

        dataallfactors = dataallfactors.drop(i)
        datafiltweek = datafiltweek.drop(i)
        continue

## section where regression is performed

# create dataframe to include all regression betas for each stock
regression_values = pd.DataFrame()
regression_values['regression parameters'] = 'b1','b2','b3','b4'


datafiltweek = datafiltweek.fillna(value=1)

# add column name to weekly dates
dataallfactors.columns = ['weekly dates','Mkt-RF', 'SMB', 'HML','RF','Sentiment factor']

for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # create copies of dataframes to be modified when certain stock returns aren't available
    datafiltweekmod = datafiltweek.copy()
    dataallfactorsmod = dataallfactors.copy()

    # loop to remove rows where sentiment factor doesn't exist yet
    for j in datafiltweek.index:

        if datafiltweek.loc[j,'weekly return of Stock '+str(stock_name)] == 1:
                
            dataallfactorsmod = dataallfactorsmod.drop(j)
            datafiltweekmod = datafiltweekmod.drop(j)

    X = dataallfactorsmod[['Mkt-RF', 'SMB', 'HML','Sentiment factor']]
    y = (datafiltweekmod['weekly return of Stock '+str(stock_name)]-1) - dataallfactorsmod['RF']
    ff_model = sm.OLS(y, X).fit()
    print(ff_model.summary())
    b1, b2, b3,b4 = ff_model.params

    for i in range(4):
        regression_values.loc[i,'regression values for Stock '+str(stock_name)] = ff_model.params[i]


regression_values = regression_values.transpose()

regression_values

regression_values.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Regression//All_Stocks_Regression.csv')

## section to find the reconstructed time series
# create new dataframe to store the reconstructed time series
reconstructedtimeseries = pd.DataFrame()

reconstructedtimeseries['weekly dates']= dataallfactors['weekly dates']
 


# loop to create time series for each stock
for i in stock_list:
    stock_name = str(i)

    for j in dataallfactors.index:
        reconstructedtimeseries.loc[j,'New time series for Stock '+ str(stock_name)] = regression_values.loc['regression values for Stock ' + str(stock_name),0]*dataallfactors.loc[j,'Mkt-RF']+regression_values.loc['regression values for Stock ' + str(stock_name),1]*dataallfactors.loc[j,'SMB']+regression_values.loc['regression values for Stock ' + str(stock_name),2]*dataallfactors.loc[j,'HML']+regression_values.loc['regression values for Stock ' + str(stock_name),3]*dataallfactors.loc[j,'Sentiment factor']

reconstructedtimeseries.mean()

datafiltweek.mean()

## compare new time series with old time series to find residuals
# create new dataframe to store the residual time series
residualtimeseries = pd.DataFrame()

residualtimeseries['weekly dates'] = dataallfactors['weekly dates']

for i in stock_list:
    stock_name = str(i)

    residualtimeseries['Residual time series of Stock '+str(stock_name)] = (reconstructedtimeseries['New time series for Stock '+ str(stock_name)]+1)-datafiltweek['weekly return of Stock '+str(stock_name)]

residualtimeseries

residualtimeseries.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Residual_time_series//All_Stocks_Residual_time_series.csv')

######### section where regression is performed without the sentiment factor

# create dataframe for regression values of each stock without sentiment factor
regression_values_minussentiment = pd.DataFrame()
regression_values_minussentiment['regression parameters'] = 'b1','b2','b3'



# regression loop for each stock without the sentiment factor
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # create copies of dataframes to be modified when certain stock returns aren't available
    datafiltweekmod = datafiltweek.copy()
    dataallfactorsmod = dataallfactors.copy()

    # loop to remove rows where sentiment factor doesn't exist yet
    for j in datafiltweek.index:

        if datafiltweek.loc[j,'weekly return of Stock '+str(stock_name)] == 1:
                
            dataallfactorsmod = dataallfactorsmod.drop(j)
            datafiltweekmod = datafiltweekmod.drop(j)

    X = dataallfactorsmod[['Mkt-RF', 'SMB', 'HML']]
    y = (datafiltweekmod['weekly return of Stock '+str(stock_name)]-1) - dataallfactorsmod['RF']
    ff_model = sm.OLS(y, X).fit()
    print(ff_model.summary())
    b1, b2, b3 = ff_model.params

    for i in range(3):
        regression_values_minussentiment.loc[i,'regression values for Stock '+str(stock_name)] = ff_model.params[i]


regression_values_minussentiment = regression_values_minussentiment.transpose()

regression_values
regression_values_minussentiment

regression_values_minussentiment.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Regression//All_Stocks_Regression_minussentiment.csv')

## section to find the reconstructed time series without sentiment factor
# create new dataframe to store the reconstructed time series
reconstructedtimeseries_minussentiment = pd.DataFrame()

reconstructedtimeseries_minussentiment['weekly dates']= dataallfactors['weekly dates']
 


# loop to create time series for each stock
for i in stock_list:
    stock_name = str(i)

    for j in dataallfactors.index:
        reconstructedtimeseries_minussentiment.loc[j,'New time series for Stock '+ str(stock_name)] = regression_values_minussentiment.loc['regression values for Stock ' + str(stock_name),0]*dataallfactors.loc[j,'Mkt-RF']+regression_values_minussentiment.loc['regression values for Stock ' + str(stock_name),1]*dataallfactors.loc[j,'SMB']+regression_values_minussentiment.loc['regression values for Stock ' + str(stock_name),2]*dataallfactors.loc[j,'HML']

reconstructedtimeseries_minussentiment.mean()

datafiltweek.mean()

## compare new time series with old time series to find residuals without sentiment factor
# create new dataframe to store the residual time series
residualtimeseries_minussentiment = pd.DataFrame()

residualtimeseries_minussentiment['weekly dates'] = dataallfactors['weekly dates']

for i in stock_list:
    stock_name = str(i)

    residualtimeseries_minussentiment['Residual time series of Stock '+str(stock_name)] = (reconstructedtimeseries_minussentiment['New time series for Stock '+ str(stock_name)]+1)-datafiltweek['weekly return of Stock '+str(stock_name)]

residualtimeseries
residualtimeseries_minussentiment

residualtimeseries_minussentiment.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Residual_time_series//All_Stocks_Residual_time_series_minussentiment.csv')
