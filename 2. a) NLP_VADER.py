
# imports needed for VADER

from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt

##########
# Stock name to process
# stock_name = 'BIIB'

##########

# create empty list for storing scores
# sentiment_scores = []

# simplification
analyser = SentimentIntensityAnalyzer()
 
# def to return scores 
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    # print("{:-<40} {}".format(sentence, str(score)))
    return score

# place holder for big stocks : 'AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU'
# place holder for other stocks : 'AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
## loop to scrape all stocks above
stock_list = ['KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','CEG','VRSN','SWKS','DOCU','SPLK','OKTA'
]



for i in stock_list:
    stock_name = str(i)
    print(stock_name)

# import file containing comments
    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_wsb_scraped_comments//'+str(stock_name)+'_wsb_comments.csv')

# old simple technique 


# loop to perform NLP with VADER and storing scores into the list with no time frame restriction
 # for i in range(100):
  # result = sentiment_analyzer_scores(data['body'][i])
   # print(result)
  # result = result.get("compound")
   # print(result)
  # sentiment_scores.append(result)

# sentiment_scores

# finding mean score 
# sentiment_scores_mean = statistics.mean(sentiment_scores)

# print(sentiment_scores_mean)


# delimit the time frames on a weekly basis

# first time frame
    first_date = int(dt.datetime(2010,1,4,0,0).timestamp())

# create list for epoch time frames
    weeklytimeframes_list = []

# add all the weekly epoch time frames to the list
    for i in range(750):
        date = first_date+i*604800
        weeklytimeframes_list.append(date)

# testing
# weeklytimeframes_list[0]
# print(len(weeklytimeframes_list))

# create empty list for storing scores
    df = pd.DataFrame(np.nan,index=np.arange(1,len(data)),columns=np.arange(750))
    weekly_average_sentiment_scores = []

# create loop to perform NLP with VADER store sentiment scores on a weekly time frame
    for j in range(750):
        weekly_sentiment_scores = []
        for i in range(len(data)):
            if data['created_utc'][i] > weeklytimeframes_list[j]:
                if data['created_utc'][i] < weeklytimeframes_list[j+1]:
                    result = sentiment_analyzer_scores(data['body'][i])
                    # print(result)
                    result = result.get("compound")
                    # print(result)
                    weekly_sentiment_scores.append(result)
    # print(weekly_sentiment_scores)
        df[j] = pd.Series(weekly_sentiment_scores)
    # if weekly_sentiment_scores != []:
       # weekly_average_sentiment_scores[j] = statistics.mean(weekly_sentiment_scores)       

# store all of the scores by week
    df.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_VADER_scores//'+str(stock_name)+'_VADER_scores.csv')

# create list containing average weekly sentiment
    weekly_sentiment_scores_mean = []
# create list containing length of weekly sentiment
    weekly_sentiment_scores_count = []

# create loop to aggregate VADER sentiment scores on a weekly basis
    for i in range(750):
        weekly_result_mean = np.nanmean(df[i])
        weekly_result_count = np.count_nonzero(~np.isnan(df[i]))
        weekly_sentiment_scores_mean.append(weekly_result_mean)
        weekly_sentiment_scores_count.append(weekly_result_count)

# store the values above
    weekly_sentiment_scores_mean
    weekly_sentiment_scores_count

# merge both lists together
    weekly_lists_merged = []    
    weekly_lists_merged.append(weekly_sentiment_scores_mean)
    weekly_lists_merged.append(weekly_sentiment_scores_count)

# create a new Dataframe for these results
    dftot = pd.DataFrame()
    dftot = pd.DataFrame(weekly_lists_merged).transpose()
    dftot.columns = ['mean of Stock '+str(stock_name),'count of Stock '+str(stock_name)]

    # dftot

# store the average scores and count in seperate file
    dftot.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_summarised_VADER_scores//'+str(stock_name)+'_summarised_VADER_scores.csv')

