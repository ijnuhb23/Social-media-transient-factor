
# imports needed for VADER
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp

# import list of active stocks in portfolio
sector_list = ['Commercial services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
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

    # normalise and standardise the mean scores columns of each stock
    datanorm['mean of '+str(sector_name)] = (datanorm['mean of '+str(sector_name)] - datanorm['mean of '+str(sector_name)].mean())/(datanorm['mean of '+str(sector_name)].std())

    # add the count columns of each stock
    datanorm['count of '+str(sector_name)] = datasector['count of '+str(sector_name)]

    # replace to 0 values in the count columns by Nan 
    datanorm['count of '+str(sector_name)] = datanorm['count of '+str(sector_name)].replace(0,np.nan)

    # normalise and standardise the count columns of each stock
    datanorm['count of '+str(sector_name)] = (datanorm['count of '+str(sector_name)] - datanorm['count of '+str(sector_name)].mean())/(datanorm['count of '+str(sector_name)].std())+1

    # multiply both mean score with count to normalise both elements together
    datanorm['count multiplied with mean score of '+str(sector_name)] = datanorm['count of '+str(sector_name)]*datanorm['mean of '+str(sector_name)]

    # datatot

# save the dataframe into a csv file
datanorm.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_sectors_norm//All_sectors_norm.csv')

# section to graph and test standardised data
#datanorm.plot( y=["mean of Consumer Durables"],
#        kind="line", figsize=(10, 10))

#mp.show()

#datanorm.plot( y=["count of Consumer Durables"],
#        kind="line", figsize=(10, 10))

#mp.show()

#datanorm.plot( y=["count multiplied with mean score of Consumer Durables"],
#        kind="line", figsize=(10, 10))

# mp.show()

# create new dataframe with final normalised scores
datanormfinal = pd.DataFrame()

for i in sector_list:
    sector_name = str(i)
    print(sector_name)

    # retrieve final score from previous dataframe and store it in new dataframe
    datanormfinal['final score of '+str(sector_name)] = datanorm['count multiplied with mean score of '+str(sector_name)]

# section to sort and rank best sectors scores and worst scores on a weekly time frame
# transpose data to make manipulations easier
datanormfinal = datanormfinal.transpose()

# rank the scores : 1 being the lowest and highest value equal to highest score
datanormfinal = datanormfinal.rank(method='dense')

datanormfinal.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_sectors_ranked_by_norm_social_scores//All_Stock_sectors_ranked_by_norm_social_scores.csv')

# find the max in each column
max = datanormfinal.max()

# replace empty values by 0 to make next step easier
datanormfinal = datanormfinal.replace(np.nan,0)

# loop to identify which value to long, based on the highest rank
for i in datanormfinal:
    
    # condition to filter timeframes where there are no scores
    if datanormfinal[i].all() == 0:
        if max[i] == 1:
            datanormfinal[i] = datanormfinal[i].replace(max[i],'nothing')
        datanormfinal[i] = datanormfinal[i].replace(max[i],'long')

# replace all of the lowest ranks with short
datanormfinal = datanormfinal.replace([1],'short')
# remove the 0 rank, since they aren't useful
datanormfinal = datanormfinal.replace(0,np.nan)

datanormfinal = datanormfinal.transpose()

# save the results
datanormfinal.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sectors_to_long_short//All_Sectors_to_long_short.csv')
