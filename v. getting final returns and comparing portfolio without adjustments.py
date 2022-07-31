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
# datalong_short = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short//All_Sectors_to_long_short.csv')

# import the long short positions per sector with inverse vol application
datalong_short = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short_inv_vol//All_Sectors_to_long_short_inv_vol.csv')

# import the returns per sector
datareturns = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_Returns//Quarterly_All_StockweightedSectors_Returns.csv')

# create new dataframe to include 
datareturns_long_short = pd.DataFrame()

# copy the returns table
datareturns_long_short = datareturns

# put the weekly dates in the long short table
datalong_short['weekly dates'] = datareturns_long_short['weekly dates']

#process to increase exposure to best sentiment scores and reduce exposure to worst sentiment scores
for i in range(0,len(datalong_short.index)-1):
    for j in range(0,len(datalong_short.columns)-1):
        if datalong_short.loc[i][j] == 'long' or datalong_short.loc[i][j] == 'short':
            if datalong_short.loc[i][j] == 'long':
                datareturns_long_short.iat[i+1,j+14] = 1.1*datareturns.iat[i+1,j+14]
            if datalong_short.loc[i][j] == 'short':
                datareturns_long_short.iat[i+1,j+14] = datareturns.iat[i+1,j+14]/1.1

# reset the Total Market Cap 
datareturns_long_short['Total Market Cap'] = 0
# recalculate total market cap and remove the percentage value of 1 (not optimal)
datareturns_long_short['Total Market Cap'] = datareturns_long_short.filter(regex='Market Cap').sum(axis=1)-1

sector_list = ['Commercial Services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
,'Health Technology','Producer Manufacturing','Retail Trade','Technology Services','Transportation','Utilities']

# loop to recalculate the percentage of total market cap for each sector
for i in sector_list:
    sector_name = str(i)
    # print(sector_name)

    # recalculate the percentage of total market cap for each sector
    datareturns_long_short[str(sector_name)+' percentage of Total Market Cap'] = datareturns_long_short[str(sector_name)+' Market Cap']/datareturns_long_short['Total Market Cap']

## section to calculate final return for sentiment adjusted portfolio
# loop to multiply percentage of Total Market Cap of each sector with the 
for i in sector_list:
    sector_name = str(i)

    datareturns_long_short[str(sector_name)+' percentage of Total Market Cap multiplied with its return'] = datareturns_long_short[str(sector_name)+' percentage of Total Market Cap']*datareturns_long_short[str(sector_name)+' returns']

# add all percentages together 
datareturns_long_short['Quarterly Return'] = datareturns_long_short.filter(regex='percentage of Total Market Cap multiplied with its return').sum(axis=1)

# show final return over the whole examined period
datareturns_long_short['Total Return'] = datareturns_long_short['Quarterly Return'].product()

# show cumulative quarterly return
# create a column for the cumulative return
datareturns_long_short['Cumulative Return'] = 1

# loop to calculate cumulative quarterly return
for i in datareturns_long_short.index:
    if i == 0:
        datareturns_long_short.loc[i,'Cumulative Return'] = datareturns_long_short.loc[i,'Quarterly Return']
    if i > 0 :
        datareturns_long_short.loc[i,'Cumulative Return'] = datareturns_long_short.loc[i,'Quarterly Return']*datareturns_long_short.loc[i-1,'Cumulative Return']



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

for i in datareturns_long_short.index:
   datareturns_long_short.loc[i,'Difference between both portfolios'] = datareturns_long_short.loc[i,'Cumulative Return'] - datareturns.loc[i,'Cumulative Return']


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

# add second cumulative returns to plot
color = 'tab:green'
ax2.plot(t,datareturns_long_short["Cumulative Return"], color = color)
fig.suptitle(' Comparing cumulative returns of stock weighted portfolio with sentiment weighting (green) with untouched portfolio (blue)')

mp.show()


fig, ax3 = mp.subplots()

color = 'tab:red'
ax3.plot(t,datareturns_long_short["Difference between both portfolios"], color = color)
fig.suptitle('Difference of cumulative returns between both stock weighted portfolios')
ax3.set_xlabel('quarter')
ax3.set_ylabel('cumulative return')



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


plotdata = pd.DataFrame(datareturns_long_short[['Commercial Services returns','Communications returns','Consumer Durables returns'
,'Consumer Non Durables returns','Consumer Services returns','Distribution Services returns','Electronic Technology returns'
,'Health Technology returns','Producer Manufacturing returns','Retail Trade returns','Technology Services returns','Transportation returns','Utilities returns']]-1,index = np.arange(30))

plotdata

plotdata.plot(kind="bar", stacked=True)

mp.show()

# plot 
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

plotdata.plot(kind="bar", stacked=True)


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

plotdata.plot(kind="bar", stacked=True)


mp.show()


## section to visually compare the influence of the weighting adjustment on all sectors on a quarterly basis

datareturns_compare = pd.DataFrame()

for i in sector_list:
    sector_name = str(i)

    datareturns_compare[str(sector_name)+ ' weighted comparison'] = datareturns_long_short[str(sector_name)+ ' percentage of Total Market Cap multiplied with its return'] - datareturns[str(sector_name)+ ' percentage of Total Market Cap multiplied with its return']

len(datareturns_compare.index)

plotdata = pd.DataFrame(datareturns_compare,index=np.arange(len(datareturns_compare.index)))
plotdata.plot(kind="bar", stacked=True)

mp.show()

