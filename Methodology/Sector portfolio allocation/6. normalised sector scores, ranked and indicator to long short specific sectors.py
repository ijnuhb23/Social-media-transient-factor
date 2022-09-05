
# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

# import list of active stocks in portfolio
sector_list = ['Commercial Services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
,'Health Technology','Producer Manufacturing','Retail Trade','Technology Services','Transportation','Utilities']

#create empty dataframe for the normalised data
datanorm = pd.DataFrame()

# get filtered sector scores
datasector = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_filt_sectors_scores//All_filt_sectors_scores.csv')

# loop to store summary data from each stock into the main dataframe
for i in sector_list:
    sector_name = str(i)
    print(sector_name)

    

    # add the mean scores columns of each stock
    datanorm['mean of '+str(sector_name)] = datasector['mean of '+str(sector_name)]

    # normalise and standardise the mean scores columns of each stock sector with a rolling window of 1 year on a weekly basis
    window = 52
    min_periods = 5
    target_column = sector_name
    roll = datanorm['mean of '+str(sector_name)].rolling(window, min_periods)
    datanorm['mean of '+str(sector_name)] = (datanorm['mean of '+str(sector_name)] - roll.mean()) / roll.std()

    # add the count columns of each stock
    # datanorm['count of '+str(sector_name)] = datasector['count of '+str(sector_name)]

    # replace to 0 values in the count columns by Nan 
    # datanorm['count of '+str(sector_name)] = datanorm['count of '+str(sector_name)].replace(0,np.nan)

    # normalise and standardise the count columns of each stock
    # datanorm['count of '+str(sector_name)] = (datanorm['count of '+str(sector_name)] - datanorm['count of '+str(sector_name)].mean())/(datanorm['count of '+str(sector_name)].std())+1

    # multiply both mean score with count to normalise both elements together
    # datanorm['count multiplied with mean score of '+str(sector_name)] = datanorm['count of '+str(sector_name)]*datanorm['mean of '+str(sector_name)]

    # datatot

# read all filtered stock data
datafiltret = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

# transpose weekly dates to the new dataframe
datanorm['weekly dates'] = datafiltret['weekly date of Stock']

# save the dataframe into a csv file
datanorm.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_sectors_norm//All_sectors_norm.csv')

# section to graph and test standardised data
datanorm.plot( y=["mean of Technology Services"],
        kind="line", figsize=(10, 10))

# mp.show()


## section to agglomerate weekly sentiment scores into quarterly scores

#create new dataframe to store quarterly scores
datanormquart = pd.DataFrame()


#create dataframe to perform interim manipulations
datanormmanip =pd.DataFrame()

# quarterly dates of targeted timeframe
quarter_dates = [
    
"2015-01-01","2015-04-01","2015-07-01","2015-10-01",
"2016-01-01","2016-04-01","2016-07-01","2016-10-01",
"2017-01-01","2017-04-01","2017-07-01","2017-10-01",
"2018-01-01","2018-04-01","2018-07-01","2018-10-01",
"2019-01-01","2019-04-01","2019-07-01","2019-10-01",
"2020-01-01","2020-04-01","2020-07-01","2020-10-01",
"2021-01-01","2021-04-01","2021-07-01","2021-10-01",
"2022-01-01","2022-04-01","2022-07-01"

]

# loop through quarterly dates apart from last value
for j in quarter_dates[:-1]:
    print(j)
    print(quarter_dates.index(j))

    # get the start date in the current loop
    start_date = quarter_dates[quarter_dates.index(j)]
    # get the end date in the current loop
    end_date = quarter_dates[quarter_dates.index(j)+1]

    # extract the quarterly interval data
    mask = (datanorm['weekly dates'] > start_date) & (datanorm['weekly dates'] <= end_date)
    datanormmanip = datanorm.loc[mask]

    # loop each column to average the weekly scores
    for i in sector_list:
        sector_name = str(i)

        # calculate the mean of average weekly scores for each sector 
        datanormmanip['mean of '+str(sector_name)] = datanormmanip['mean of '+str(sector_name)].mean()

    # only keep the first row of data, since they all contain the same values and makes it easier for appending with the other data points
    datanormmanip = datanormmanip.loc[datanormmanip.index[0]]

    # identify each quarter with the starting date of the quarter
    datanormmanip['weekly dates'] = start_date


    datanormquart = datanormquart.append(datanormmanip)

datanormquart 

# save the quarterly scores
datanormquart.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_Stock_sectors_by_norm_social_scores//Quarterly_Stock_sectors_by_norm_social_scores.csv')

# section to sort and rank best sectors scores and worst scores on a weekly time frame
# transpose data to make manipulations easier

# rank the scores : 1 being the lowest and highest value equal to highest score
datanormquart = datanormquart.rank(axis=1,method='dense')

datanormquart.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_Stock_sectors_ranked_by_norm_social_scores//Quarterly_Stock_sectors_ranked_by_norm_social_scores.csv')

# find the max in each column
max = datanormquart.max(axis = 1)

# replace empty values by 0 to make next step easier
datanormquart = datanormquart.replace(np.nan,0)

# loop to identify which value to long, based on the highest rank by row
for i in datanormquart.index:

    # loop by column to check every single cell in dataframe
    for j in datanormquart:

        # condition to check if value in cell is equal to the maximum in the specific row
        if datanormquart.loc[i][j] == max[i]:
            # set a high value to indicate which value in row is highest score
            datanormquart.loc[i][j] = 9999

# replace the maximum values found previously with the word long
datanormquart = datanormquart.replace([9999],'long')
# replace all of the lowest ranks with short
datanormquart = datanormquart.replace([1],'short')
# remove the 0 rank, since they aren't useful
datanormquart = datanormquart.replace(0,np.nan)

datanormquart

# save the results
datanormquart.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short//All_Sectors_to_long_short.csv')
