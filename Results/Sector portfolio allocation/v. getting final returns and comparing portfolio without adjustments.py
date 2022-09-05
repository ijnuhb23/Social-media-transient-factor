# imports needed for VADER
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

# import the long short positions per sector
datalong_short = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short//All_Sectors_to_long_short.csv')

# import the long short positions per sector with inverse vol application
# datalong_short = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short_inv_vol//All_Sectors_to_long_short_inv_vol.csv')

# import the returns per sector
datareturns = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_Returns//Quarterly_All_StockweightedSectors_Returns.csv')

# create new dataframe to include 
datareturns_long_short = pd.DataFrame()
datareturns_long_short_15 = pd.DataFrame()
datareturns_long_short_20 = pd.DataFrame()

# create new dataframe for modified sector weights
datareturns_long_short_only = pd.DataFrame()


# deep copy the returns table
datareturns_long_short = datareturns.copy()
datareturns_long_short_15 = datareturns.copy()
datareturns_long_short_20 = datareturns.copy()
datareturns_long_short_only = datareturns.copy()

# put the weekly dates in the long short table
datalong_short['weekly dates'] = datareturns_long_short['weekly dates'].copy()
datareturns_long_short_only['weekly dates'] = datareturns_long_short['weekly dates'].copy()

# rename certain columns of the long_short_only dataframe
datareturns_long_short_only = datareturns_long_short_only.rename(columns = {'Commercial Services returns':'Long Sector return', 'Communications returns': 'Short Sector return'}, inplace = False)

#process to increase exposure to best sentiment scores and reduce exposure to worst sentiment scores
for i in range(0,len(datalong_short.index)-1):
    for j in range(0,len(datalong_short.columns)-1):
        if datalong_short.loc[i][j] == 'long' or datalong_short.loc[i][j] == 'short':
            if datalong_short.loc[i][j] == 'long':
                datareturns_long_short.iat[i+1,j+14] = 1.1*datareturns.iat[i+1,j+14]
                datareturns_long_short_15.iat[i+1,j+14] = 1.5*datareturns.iat[i+1,j+14]      
                datareturns_long_short_20.iat[i+1,j+14] = 2*datareturns.iat[i+1,j+14]                            
            if datalong_short.loc[i][j] == 'short':
                datareturns_long_short.iat[i+1,j+14] = datareturns.iat[i+1,j+14]*0.9
                datareturns_long_short_15.iat[i+1,j+14] = datareturns.iat[i+1,j+14]*0.5  
                datareturns_long_short_20.iat[i+1,j+14] = datareturns.iat[i+1,j+14]*0

#process to extract sector returns of long and short exposure sectors
for i in range(0,len(datalong_short.index)-1):
    for j in range(0,len(datalong_short.columns)-1):
        if datalong_short.loc[i][j] == 'long' or datalong_short.loc[i][j] == 'short':
            if datalong_short.loc[i][j] == 'long':
                datareturns_long_short_only.iat[i+1,2] = datareturns.iat[i+1,j+1]
            if datalong_short.loc[i][j] == 'short':
                datareturns_long_short_only.iat[i+1,3] = datareturns.iat[i+1,j+1]

# reset a value in the short sector return
datareturns_long_short_only.iat[0,3] = 1

# find the mean of the long sector only portfolio and the short sector only portfolio

datareturns_long_short_only['Long Sector mean'] = datareturns_long_short_only['Long Sector return'].mean()
datareturns_long_short_only['Short Sector mean'] = datareturns_long_short_only['Short Sector return'].mean()

datareturns_long_short_only

# reset the Total Market Cap 
datareturns_long_short['Total Market Cap'] = 0
datareturns_long_short_15['Total Market Cap'] = 0
datareturns_long_short_20['Total Market Cap'] = 0 

# recalculate total market cap and remove the percentage value of 1 (not optimal)
datareturns_long_short['Total Market Cap'] = datareturns_long_short.filter(regex='Market Cap').sum(axis=1)-1
datareturns_long_short_15['Total Market Cap'] = datareturns_long_short_15.filter(regex='Market Cap').sum(axis=1)-1
datareturns_long_short_20['Total Market Cap'] = datareturns_long_short_20.filter(regex='Market Cap').sum(axis=1)-1

sector_list = ['Commercial Services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
,'Health Technology','Producer Manufacturing','Retail Trade','Technology Services','Transportation','Utilities']

# loop to recalculate the percentage of total market cap for each sector
for i in sector_list:
    sector_name = str(i)
    # print(sector_name)

    # recalculate the percentage of total market cap for each sector
    datareturns_long_short[str(sector_name)+' percentage of Total Market Cap'] = datareturns_long_short[str(sector_name)+' Market Cap']/datareturns_long_short['Total Market Cap']
    datareturns_long_short_15[str(sector_name)+' percentage of Total Market Cap'] = datareturns_long_short_15[str(sector_name)+' Market Cap']/datareturns_long_short_15['Total Market Cap']
    datareturns_long_short_20[str(sector_name)+' percentage of Total Market Cap'] = datareturns_long_short_20[str(sector_name)+' Market Cap']/datareturns_long_short_20['Total Market Cap']        

## section to calculate final return for sentiment adjusted portfolio
# loop to multiply percentage of Total Market Cap of each sector with the 
for i in sector_list:
    sector_name = str(i)

    datareturns_long_short[str(sector_name)+' percentage of Total Market Cap multiplied with its return'] = datareturns_long_short[str(sector_name)+' percentage of Total Market Cap']*datareturns_long_short[str(sector_name)+' returns']
    datareturns_long_short_15[str(sector_name)+' percentage of Total Market Cap multiplied with its return'] = datareturns_long_short_15[str(sector_name)+' percentage of Total Market Cap']*datareturns_long_short_15[str(sector_name)+' returns']
    datareturns_long_short_20[str(sector_name)+' percentage of Total Market Cap multiplied with its return'] = datareturns_long_short_20[str(sector_name)+' percentage of Total Market Cap']*datareturns_long_short_20[str(sector_name)+' returns']


# add all percentages together 
datareturns_long_short['Quarterly Return'] = datareturns_long_short.filter(regex='percentage of Total Market Cap multiplied with its return').sum(axis=1)
datareturns_long_short_15['Quarterly Return'] = datareturns_long_short_15.filter(regex='percentage of Total Market Cap multiplied with its return').sum(axis=1)
datareturns_long_short_20['Quarterly Return'] = datareturns_long_short_20.filter(regex='percentage of Total Market Cap multiplied with its return').sum(axis=1)

# show final return over the whole examined period
datareturns_long_short['Total Return'] = datareturns_long_short['Quarterly Return'].product()
datareturns_long_short_15['Total Return'] = datareturns_long_short_15['Quarterly Return'].product()
datareturns_long_short_20['Total Return'] = datareturns_long_short_20['Quarterly Return'].product()

# show cumulative quarterly return
# create a column for the cumulative return
datareturns_long_short['Cumulative Return'] = 1
datareturns_long_short_15['Cumulative Return'] = 1
datareturns_long_short_20['Cumulative Return'] = 1

# loop to calculate cumulative quarterly return
for i in datareturns_long_short.index:
    if i == 0:
        datareturns_long_short.loc[i,'Cumulative Return'] = datareturns_long_short.loc[i,'Quarterly Return']
        datareturns_long_short_15.loc[i,'Cumulative Return'] = datareturns_long_short_15.loc[i,'Quarterly Return']
        datareturns_long_short_20.loc[i,'Cumulative Return'] = datareturns_long_short_20.loc[i,'Quarterly Return']
    if i > 0 :
        datareturns_long_short.loc[i,'Cumulative Return'] = datareturns_long_short.loc[i,'Quarterly Return']*datareturns_long_short.loc[i-1,'Cumulative Return']
        datareturns_long_short_15.loc[i,'Cumulative Return'] = datareturns_long_short_15.loc[i,'Quarterly Return']*datareturns_long_short_15.loc[i-1,'Cumulative Return']
        datareturns_long_short_20.loc[i,'Cumulative Return'] = datareturns_long_short_20.loc[i,'Quarterly Return']*datareturns_long_short_20.loc[i-1,'Cumulative Return']        



## section to calculate final return for untouched portfolio for comparisons purposes

# import the returns per sector once again
datareturns = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_Returns//Quarterly_All_StockweightedSectors_Returns.csv')

# loop to multiply percentage of Total Market Cap of each sector with the 
for i in sector_list:
    sector_name = str(i)

    datareturns[str(sector_name)+' percentage of Total Market Cap multiplied with its return'] = datareturns[str(sector_name)+' percentage of Total Market Cap']*datareturns[str(sector_name)+' returns']

# add all percentages together 
datareturns['Quarterly Return'] = datareturns.filter(regex='percentage of Total Market Cap multiplied with its return').sum(axis=1)

# show final return over the whole examined period
datareturns['Total Return'] = datareturns['Quarterly Return'].product()

# show cumulative quarterly return
# create a column for the cumulative return
datareturns['Cumulative Return'] = 1

# loop to calculate cumulative quarterly return
for i in datareturns.index:
    if i == 0:
        datareturns.loc[i,'Cumulative Return'] = datareturns.loc[i,'Quarterly Return']
    if i > 0 :
        datareturns.loc[i,'Cumulative Return'] = datareturns.loc[i,'Quarterly Return']*datareturns.loc[i-1,'Cumulative Return']

# save both results
datareturns_long_short.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_Adjusted_Returns//Quarterly_All_StockweightedSectors_Adjusted_Returns.csv')

datareturns.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_NonAdjusted_Returns//Quarterly_All_StockweightedSectors_NonAdjusted_Returns.csv')

## section to show slight improvment by using social media sentiment
datareturns
datareturns_long_short

## section to compare the quarterly returns between both portfolios
# create a new column for the data
datareturns_long_short['Difference between both portfolios'] = 1
datareturns_long_short_15['Difference between both portfolios'] = 1
datareturns_long_short_20['Difference between both portfolios'] = 1

for i in datareturns_long_short.index:
   datareturns_long_short.loc[i,'Difference between both portfolios'] = datareturns_long_short.loc[i,'Cumulative Return'] - datareturns.loc[i,'Cumulative Return']
   datareturns_long_short_15.loc[i,'Difference between both portfolios'] = datareturns_long_short_15.loc[i,'Cumulative Return'] - datareturns.loc[i,'Cumulative Return']
   datareturns_long_short_20.loc[i,'Difference between both portfolios'] = datareturns_long_short_20.loc[i,'Cumulative Return'] - datareturns.loc[i,'Cumulative Return']

# section to calculate maximum drawdown from beginning of investment period

# https://medium.com/cloudcraftz/measuring-maximum-drawdown-and-its-python-implementation-99a3963e158f

Roll_Max = datareturns_long_short['Cumulative Return'].rolling(100,min_periods=1).max()
Drawdown = datareturns_long_short['Cumulative Return']/Roll_Max - 1
Max_Drawdown = Drawdown.rolling(100,min_periods=1).min()

Roll_Max = datareturns_long_short_15['Cumulative Return'].rolling(100,min_periods=1).max()
Drawdown = datareturns_long_short_15['Cumulative Return']/Roll_Max - 1
Max_Drawdown_15 = Drawdown.rolling(100,min_periods=1).min()

Roll_Max = datareturns_long_short_20['Cumulative Return'].rolling(100,min_periods=1).max()
Drawdown = datareturns_long_short_20['Cumulative Return']/Roll_Max - 1
Max_Drawdown_20 = Drawdown.rolling(100,min_periods=1).min()

Max_Drawdown.plot(color = 'green')
Max_Drawdown_15.plot(color = 'red') 
Max_Drawdown_20.plot(color = 'brown')
mp.suptitle(' Maximum drawdown over the time frame') 
mp.xlabel('quarter')
mp.ylabel('percentage drawdown')

mp.show()

# section to calculate average returns

datareturns['Expected Return'] = (datareturns['Quarterly Return']-1).mean()
datareturns_long_short['Expected Return'] = (datareturns_long_short['Quarterly Return']-1).mean()
datareturns_long_short_15['Expected Return'] = (datareturns_long_short_15['Quarterly Return']-1).mean()
datareturns_long_short_20['Expected Return'] = (datareturns_long_short_20['Quarterly Return']-1).mean()

datareturns['Expected Return']
datareturns_long_short['Expected Return']
datareturns_long_short_15['Expected Return']
datareturns_long_short_20['Expected Return']

# section to calculate standard deviation of returns

datareturns['Standard deviation of Returns'] = (datareturns['Quarterly Return']-1).std()
datareturns_long_short['Standard deviation of Returns'] = (datareturns_long_short['Quarterly Return']-1).std()
datareturns_long_short_15['Standard deviation of Returns'] = (datareturns_long_short_15['Quarterly Return']-1).std()
datareturns_long_short_20['Standard deviation of Returns'] = (datareturns_long_short_20['Quarterly Return']-1).std()

datareturns['Standard deviation of Returns']
datareturns_long_short['Standard deviation of Returns']
datareturns_long_short_15['Standard deviation of Returns']
datareturns_long_short_20['Standard deviation of Returns']

# to compile best standard deviation portfolio composition

# best standard portfolio is at 65% increase/decrease with 0.117883



# secltion to calculate best sharpe ratio

datareturns['Sharpe Ratio'] = datareturns['Expected Return']/datareturns['Standard deviation of Returns']
datareturns_long_short['Sharpe Ratio'] = datareturns_long_short['Expected Return']/datareturns_long_short['Standard deviation of Returns']
datareturns_long_short_15['Sharpe Ratio'] = datareturns_long_short_15['Expected Return']/datareturns_long_short_15['Standard deviation of Returns']
datareturns_long_short_20['Sharpe Ratio'] = datareturns_long_short_20['Expected Return']/datareturns_long_short_20['Standard deviation of Returns']

datareturns['Sharpe Ratio']
datareturns_long_short['Sharpe Ratio']
datareturns_long_short_15['Sharpe Ratio']
datareturns_long_short_20['Sharpe Ratio']


## section to display improvment in cumulative return
# set x range
t = np.arange(0.0,30,1)

fig, ax1 = mp.subplots()

# add first cumulative returns to plot
color = 'tab:blue'
ax1.plot(t,datareturns["Cumulative Return"], color = color)
ax1.set_xlabel('quarter')
ax1.set_ylabel('cumulative return')



# merge both axes for both dataframes
ax2 = ax1.twinx()
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax3 = ax1.twiny()
ax4 = ax1.twinx()
ax4 = ax1.twiny()

# add second cumulative returns to plot
color = 'tab:green'
ax2.plot(t,datareturns_long_short["Cumulative Return"], color = color)
color = 'tab:red'
ax3.plot(t,datareturns_long_short_15["Cumulative Return"], color = color)
color = 'tab:brown'
ax4.plot(t,datareturns_long_short_20["Cumulative Return"], color = color)

fig.suptitle(' Comparing cumulative returns of stock weighted portfolio with sentiment weighting (all except blue) with untouched portfolio (blue)')

mp.show()



fig, ax1 = mp.subplots()

# add first cumulative returns comparison to plot
color = 'tab:green'
ax1.plot(t,datareturns_long_short["Difference between both portfolios"], color = color)
fig.suptitle('Difference of cumulative returns between stock weighted portfolios')
ax1.set_xlabel('quarter')
ax1.set_ylabel('cumulative return')

# merge both axes for both dataframes
ax2 = ax1.twinx()
ax2 = ax1.twiny()
ax3 = ax1.twinx()
ax3 = ax1.twiny()

# add other cumulative returns comparisons
color = 'tab:red'
ax2.plot(t,datareturns_long_short_15["Difference between both portfolios"], color = color)
color = 'tab:brown'
ax3.plot(t,datareturns_long_short_20["Difference between both portfolios"], color = color)

mp.show()


# create a stacked bar chart to show divergence between sectors every quarter
N=30
ind = np.arange(N)
fig, ax = mp.subplots()


ax.bar(ind, datareturns_long_short['Commercial Services returns']-1, width= 0.35, label='Commercial Services returns')
ax.bar(ind, datareturns_long_short['Communications returns']-1, width= 0.35, label='Communications returns', bottom = datareturns_long_short['Commercial Services returns']-1)
ax.bar(ind, datareturns_long_short['Consumer Durables returns']-1, width= 0.35, label='Consumer Durables returns', bottom = datareturns_long_short['Commercial Services returns']-1)

ax.legend()

mp.show()

# plot that shows the returns of each sector for each quarter
plotdata = pd.DataFrame(datareturns_long_short[['Commercial Services returns','Communications returns','Consumer Durables returns'
,'Consumer Non Durables returns','Consumer Services returns','Distribution Services returns','Electronic Technology returns'
,'Health Technology returns','Producer Manufacturing returns','Retail Trade returns','Technology Services returns','Transportation returns','Utilities returns']]-1,index = np.arange(30))
plotdata
plotdata.plot(kind="bar", stacked=True,color = ['red','blue','green','orange','pink','purple','brown','yellow','black','grey','cyan','gold','silver'])
mp.title(' Stacked returns of each sector for each quarter')
mp.legend(fontsize=4)
mp.xlabel('quarter')
mp.ylabel('quarterly return of each portfolio stacked')
mp.show()

# plot that shows the sector returns of the long shorted sectors for each quarter
plotdata = pd.DataFrame(datareturns_long_short_only[['Long Sector return','Short Sector return']]-1,index = np.arange(30))
plotdata
plotdata.plot(kind="bar", stacked=True)
mp.title(' Comparing chosen long and shorted sectors returns every quarter')
mp.legend(fontsize=7)
mp.legend(title='mean long sector: 0.067065; mean short sector: 0.048835' , title_fontsize=7)
mp.xlabel('quarter')
mp.ylabel('quarterly return of chosen sector')
mp.show()


# plot that shows the weigthed return of each sector in the total return for each quarter
plotdata = pd.DataFrame(datareturns_long_short[['Commercial Services percentage of Total Market Cap multiplied with its return','Communications percentage of Total Market Cap multiplied with its return'
,'Consumer Durables percentage of Total Market Cap multiplied with its return'
,'Consumer Non Durables percentage of Total Market Cap multiplied with its return'
,'Consumer Services percentage of Total Market Cap multiplied with its return'
,'Distribution Services percentage of Total Market Cap multiplied with its return'
,'Electronic Technology percentage of Total Market Cap multiplied with its return'
,'Health Technology percentage of Total Market Cap multiplied with its return'
,'Producer Manufacturing percentage of Total Market Cap multiplied with its return'
,'Retail Trade percentage of Total Market Cap multiplied with its return'
,'Technology Services percentage of Total Market Cap multiplied with its return'
,'Transportation percentage of Total Market Cap multiplied with its return'
,'Utilities percentage of Total Market Cap multiplied with its return']],index = np.arange(30))

plotdata.plot(kind="bar", stacked=True,color = ['red','blue','green','orange','pink','purple','brown','yellow','black','grey','cyan','gold','silver'])
mp.legend(fontsize=4)
mp.title('Composition of quarterly return of portfolio with long short sector input')
mp.xlabel('quarter')
mp.ylabel('quarterly return of portfolio')
mp.show()


## plot that decomposes the quarterly return into sectors
plotdata = pd.DataFrame(datareturns[['Commercial Services percentage of Total Market Cap multiplied with its return','Communications percentage of Total Market Cap multiplied with its return'
,'Consumer Durables percentage of Total Market Cap multiplied with its return'
,'Consumer Non Durables percentage of Total Market Cap multiplied with its return'
,'Consumer Services percentage of Total Market Cap multiplied with its return'
,'Distribution Services percentage of Total Market Cap multiplied with its return'
,'Electronic Technology percentage of Total Market Cap multiplied with its return'
,'Health Technology percentage of Total Market Cap multiplied with its return'
,'Producer Manufacturing percentage of Total Market Cap multiplied with its return'
,'Retail Trade percentage of Total Market Cap multiplied with its return'
,'Technology Services percentage of Total Market Cap multiplied with its return'
,'Transportation percentage of Total Market Cap multiplied with its return'
,'Utilities percentage of Total Market Cap multiplied with its return']],index = np.arange(30))

plotdata.plot(kind="bar", stacked=True,color = ['red','blue','green','orange','pink','purple','brown','yellow','black','grey','cyan','gold','silver'])
mp.legend(fontsize=4)
mp.title('Composition of quarterly return of portfolio without long short sector input')
mp.xlabel('quarter')
mp.ylabel('quarterly return of portfolio')
mp.show()


## section to visually compare the influence of the weighting adjustment on all sectors on a quarterly basis

datareturns_compare = pd.DataFrame()

for i in sector_list:
    sector_name = str(i)

    datareturns_compare[str(sector_name)+ ' weighted comparison'] = datareturns_long_short[str(sector_name)+ ' percentage of Total Market Cap multiplied with its return'] - datareturns[str(sector_name)+ ' percentage of Total Market Cap multiplied with its return']

len(datareturns_compare.index)

plotdata = pd.DataFrame(datareturns_compare,index=np.arange(len(datareturns_compare.index)))
plotdata.plot(kind="bar", stacked=True,color = ['red','blue','green','orange','pink','purple','brown','yellow','black','grey','cyan','gold','silver'])
mp.legend(fontsize=10)
mp.title('Decomposed quarterly improvement on the portfolio return by sector')
mp.xlabel('quarter')
mp.ylabel('decomposed quarterly return improvement')
mp.show()

datareturns_compare['Sum of weight adjustments'] = datareturns_compare.sum(axis=1)
datareturns_compare

## section to highlight total quarterly return improvement 

plotdata = pd.DataFrame(datareturns_compare['Sum of weight adjustments'],index=np.arange(len(datareturns_compare.index)))
plotdata.plot(kind="bar", stacked=True,color = ['blue'])
mp.title('Quarterly improvement on the portfolio return')
mp.legend(fontsize=4)
mp.xlabel('quarter')
mp.ylabel('quarterly return improvement')
mp.legend(title='the total improvement on the portfolio is 0.01721' , title_fontsize=7)
mp.show()


sum = datareturns_compare['Sum of weight adjustments'].sum(axis=0)

sum

meansum = sum/28

meansum