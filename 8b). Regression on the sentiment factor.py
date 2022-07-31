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

# weekly fama-french 3 factor
datafamafactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Fama_French_Factors//F-F_Research_Data_5_Factors_2x3.csv')

dataewmasentfactor
datafamafactor

# merge both dataframes together

dataallfactors = datafamafactor.copy()

dataallfactors['Sentiment factor'] = dataewmasentfactor['weekly sentiment factor']

dataallfactors

# remove excess fama-french factors
dataallfactors = dataallfactors[:-3]

dataallfactors

dataallfactors.index



## section where regression is performed

# remove rows where sentiment factor doesn't exist yet
for i in dataallfactors.index:

    if dataallfactors.loc[i,'Sentiment factor'] == 0:

        dataallfactors = dataallfactors.drop(i)
        continue
        
    if dataallfactors.loc[i,'Sentiment factor'] >= 90:

        dataallfactors = dataallfactors.drop(i)
        continue

    if dataallfactors.loc[i,'Sentiment factor'] <= -90:

        dataallfactors = dataallfactors.drop(i)
        continue


    

# assigning the X and Y for the regression and run the regression

X = dataallfactors[['Mkt-RF', 'SMB', 'HML']] # ,'RMW','CMA'
y = dataallfactors['Sentiment factor'] - dataallfactors['RF']
ff_model = sm.OLS(y, X).fit()
print(ff_model.summary())
intercept, b1, b2, b3 = ff_model.params


regression_values = pd.DataFrame()
regression_values['regression parameters'] = 'intercept','b1','b2'

# create a dataframe to assign the regression
for i in range(3):
    regression_values.loc[i,'sentiment factor'] = ff_model.params[i]

regression_values