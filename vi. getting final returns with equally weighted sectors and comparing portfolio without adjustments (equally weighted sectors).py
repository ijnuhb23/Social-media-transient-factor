# imports needed
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

#process to increase exposure to best sentiment scores and reduce exposure on an equally weighted portfolio
for i in range(0,len(datalong_short.index)-1):
    print(i)
    for j in range(1,len(datalong_short.columns)-1):
        if i == 0:
            datareturns_long_short.iat[i,j+14] = 1
        if datalong_short.loc[i][j] == 'long' or datalong_short.loc[i][j] == 'short':
            if datalong_short.loc[i][j] == 'long':
                datareturns_long_short.iat[i+1,j+14] = 1.1
            if datalong_short.loc[i][j] == 'short':
                datareturns_long_short.iat[i+1,j+14] = 0.9
        else:
            datareturns_long_short.iat[i+1,j+14] = 1

# reset the Total Market Cap 
datareturns_long_short['Total Market Cap'] = 0
# recalculate total market cap and remove the percentage value of 1 (not optimal)
datareturns_long_short['Total Market Cap'] = datareturns_long_short.filter(regex='Market Cap').sum(axis=1)-1

sector_list = ['Commercial Services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
,'Health Technology','Producer Manufacturing','Retail Trade','Technology Services','Transportation','Utilities']

## section to calculate final return for sentiment adjusted portfolio
# loop to get return of each sector divided by the number of sectors
for i in sector_list:
    sector_name = str(i)

    datareturns_long_short[str(sector_name)+' equally weighted market cap multiplied with its return'] = datareturns_long_short[str(sector_name)+' Market Cap']*datareturns_long_short[str(sector_name)+' returns']/13

# add all percentages together 
datareturns_long_short['Quarterly Return'] = datareturns_long_short.filter(regex=' equally weighted market cap multiplied with its return').sum(axis=1)

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

#process to make an equally weighted portfolio
for i in range(0,len(datalong_short.index)-1):
    print(i)
    for j in range(1,len(datalong_short.columns)-1):
        if i == 0:
            datareturns.iat[i,j+14] = 1
            datareturns.iat[i+1,j+14] = 1
        else:
            datareturns.iat[i+1,j+14] = 1

# loop to get return of each sector divided by the number of sectors
for i in sector_list:
    sector_name = str(i)

    datareturns[str(sector_name)+' equally weighted market cap multiplied with its return'] = datareturns[str(sector_name)+' Market Cap']*datareturns[str(sector_name)+' returns']/13

# add all percentages together 
datareturns['Quarterly Return'] = datareturns.filter(regex=' equally weighted market cap multiplied with its return').sum(axis=1)

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

datareturns_long_short.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_EquallyweightedSectors_Adjusted_Returns//Quarterly_All_EquallyweightedSectors_Adjusted_Returns.csv')

datareturns.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_EquallyweightedSectors_NonAdjusted_Returns//Quarterly_All_EquallyweightedSectors_NonAdjusted_Returns.csv')
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
fig.suptitle(' Comparing cumulative returns of equally weighted portfolio with sentiment weighting (green) with untouched portfolio (blue)')



mp.show()

fig, ax3 = mp.subplots()

color = 'tab:red'
ax3.plot(t,datareturns_long_short["Difference between both portfolios"], color = color)
fig.suptitle(' Difference of cumulative returns between both equally weighted portfolios')
ax3.set_xlabel('quarter')
ax3.set_ylabel('cumulative return')


mp.show()


plotdata = pd.DataFrame(datareturns_long_short[['Commercial Services returns','Communications returns','Consumer Durables returns'
,'Consumer Non Durables returns','Consumer Services returns','Distribution Services returns','Electronic Technology returns'
,'Health Technology returns','Producer Manufacturing returns','Retail Trade returns','Technology Services returns','Transportation returns','Utilities returns']]-1,index = np.arange(30))

plotdata

plotdata.plot(kind="bar", stacked=True)

mp.show()

# plot 
plotdata = pd.DataFrame(datareturns_long_short[['Commercial Services equally weighted market cap multiplied with its return'
,'Communications equally weighted market cap multiplied with its return'
,'Consumer Durables equally weighted market cap multiplied with its return'
,'Consumer Non Durables equally weighted market cap multiplied with its return'
,'Consumer Services equally weighted market cap multiplied with its return'
,'Distribution Services equally weighted market cap multiplied with its return'
,'Electronic Technology equally weighted market cap multiplied with its return'
,'Health Technology equally weighted market cap multiplied with its return'
,'Producer Manufacturing equally weighted market cap multiplied with its return'
,'Retail Trade equally weighted market cap multiplied with its return'
,'Technology Services equally weighted market cap multiplied with its return'
,'Transportation equally weighted market cap multiplied with its return'
,'Utilities equally weighted market cap multiplied with its return']],index = np.arange(30))

plotdata.plot(kind="bar", stacked=True)


mp.show()


## plot that decomposes the quarterly return into sectors
plotdata = pd.DataFrame(datareturns[['Commercial Services equally weighted market cap multiplied with its return'
,'Communications equally weighted market cap multiplied with its return'
,'Consumer Durables equally weighted market cap multiplied with its return'
,'Consumer Non Durables equally weighted market cap multiplied with its return'
,'Consumer Services equally weighted market cap multiplied with its return'
,'Distribution Services equally weighted market cap multiplied with its return'
,'Electronic Technology equally weighted market cap multiplied with its return'
,'Health Technology equally weighted market cap multiplied with its return'
,'Producer Manufacturing equally weighted market cap multiplied with its return'
,'Retail Trade equally weighted market cap multiplied with its return'
,'Technology Services equally weighted market cap multiplied with its return'
,'Transportation equally weighted market cap multiplied with its return'
,'Utilities equally weighted market cap multiplied with its return']],index = np.arange(30))

plotdata.plot(kind="bar", stacked=True)


mp.show()


## section to visually compare the influence of the weighting adjustment on all sectors on a quarterly basis

datareturns_compare = pd.DataFrame()

for i in sector_list:
    sector_name = str(i)

    datareturns_compare[str(sector_name)+ ' weighted comparion'] = datareturns_long_short[str(sector_name)+ ' equally weighted market cap multiplied with its return'] - datareturns[str(sector_name)+ ' equally weighted market cap multiplied with its return']

plotdata = pd.DataFrame(datareturns_compare,index=np.arange(len(datareturns_compare.index)))
plotdata.plot(kind="bar", stacked=True)

mp.show()

#datareturns_compare['Sum of percentages'] = datareturns_compare.sum(axis=1)

#datareturns_compare

#plotdata = pd.DataFrame(datareturns_compare,index=np.arange(len(datareturns_compare.index)))
#plotdata.plot(kind="bar", stacked=False)

#mp.show()