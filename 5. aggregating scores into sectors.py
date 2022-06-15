# imports needed for aggregating into sectors

import datetime as dt
import statistics
from calendar import week
from cmath import nan

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# read all filtered stock data
datafilt = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_summarised_filter_VADER_scores//All_summarised_filtered_VADER_scores.csv')

# read the file containing stocks with their respective sectors
stock_sectors = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_sectors//Stock_sectors.csv')

# Create empty lists of each sector
# These sectors could be simplified if needed later
Commercial_services = []
Communications = []
Consumer_Durables = []
Consumer_Non_Durables = []
Consumer_Services = []
Distribution_Servcies = []
Electronic_Technology = []
Health_Technology = []
Producer_Manufacturing = []
Retail_Trade = []
Technology_Services = []
Transportation = []
Utilities = []

# group stocks by sector
for i in range(len(stock_sectors)):
    if stock_sectors['Sector'][i] == 'Commercial Services':
        Commercial_services.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Communications':
        Communications.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Durables':
        Consumer_Durables.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Non-Durables':
        Consumer_Non_Durables.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Services':
        Consumer_Services.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Distribution Services':
        Distribution_Servcies.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Electronic Technology':
        Electronic_Technology.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Health Technology':
        Health_Technology.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Producer Manufacturing':
        Producer_Manufacturing.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Retail Trade':
        Retail_Trade.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Technology Services':
        Technology_Services.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Transportation':
        Transportation.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Utilities':
       Utilities.append(stock_sectors['Ticker'][i])               

# create new dataframes containing the scores for each sector
Commercial_services_datafiltsector = pd.DataFrame()
Communications_datafiltsector = pd.DataFrame()
Consumer_Durables_datafiltsector = pd.DataFrame()
Consumer_Non_Durables_datafiltsector = pd.DataFrame()
Consumer_Services_datafiltsector = pd.DataFrame()
Distribution_Services_datafiltsector = pd.DataFrame()
Electronic_Technology_datafiltsector = pd.DataFrame()
Health_Technology_datafiltsector = pd.DataFrame()
Producer_Manufacturing_datafiltsector = pd.DataFrame()
Retail_Trade_datafiltsector = pd.DataFrame()
Technology_Services_datafiltsector = pd.DataFrame()
Transportation_datafiltsector = pd.DataFrame()
Utilities_datafiltsector = pd.DataFrame()

# seperate the stocks into their respective sectors to facilitate calculations in the next step and include a column multiplying the mean with the count
for i in datafilt:
    print(i)

    for j in Commercial_services:
        if str(i) == 'mean of Stock ' + j:
            Commercial_services_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Commercial_services_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Commercial_services_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Communications:
        if str(i) == 'mean of Stock ' + j:
            Communications_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Communications_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Communications_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Consumer_Durables:
        if str(i) == 'mean of Stock ' + j:
            Consumer_Durables_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Consumer_Durables_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Consumer_Durables_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Consumer_Non_Durables:
        if str(i) == 'mean of Stock ' + j:
            Consumer_Non_Durables_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Consumer_Non_Durables_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Consumer_Non_Durables_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Consumer_Services:
        if str(i) == 'mean of Stock ' + j:
            Consumer_Services_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Consumer_Services_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Consumer_Services_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Distribution_Servcies:
        if str(i) == 'mean of Stock ' + j:
            Distribution_Services_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Distribution_Services_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Distribution_Services_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]
            
    for j in Electronic_Technology:
        if str(i) == 'mean of Stock ' + j:
            Electronic_Technology_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Electronic_Technology_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Electronic_Technology_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Health_Technology:
        if str(i) == 'mean of Stock ' + j:
            Health_Technology_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Health_Technology_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Health_Technology_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Producer_Manufacturing:
        if str(i) == 'mean of Stock ' + j:
            Producer_Manufacturing_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Producer_Manufacturing_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Producer_Manufacturing_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Retail_Trade:
        if str(i) == 'mean of Stock ' + j:
            Retail_Trade_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Retail_Trade_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Retail_Trade_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Technology_Services:
        if str(i) == 'mean of Stock ' + j:
            Technology_Services_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Technology_Services_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Technology_Services_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Transportation:
        if str(i) == 'mean of Stock ' + j:
            Transportation_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Transportation_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Transportation_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

    for j in Utilities:
        if str(i) == 'mean of Stock ' + j:
            Utilities_datafiltsector['mean of Stock ' + j] = datafilt['mean of Stock ' + j]
        if str(i) == 'count of Stock ' + j:
            Utilities_datafiltsector['count of Stock ' + j] = datafilt['count of Stock ' + j]
            Utilities_datafiltsector['Mean times the count of ' + j] = datafilt['count of Stock ' + j]*datafilt['mean of Stock ' + j]

# add the count of scores for all the stocks in specific sector
Commercial_services_datafiltsector['count of whole Sector'] = Commercial_services_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Communications_datafiltsector['count of whole Sector'] = Communications_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Consumer_Durables_datafiltsector['count of whole Sector'] = Consumer_Durables_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Consumer_Non_Durables_datafiltsector['count of whole Sector'] = Consumer_Non_Durables_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Consumer_Services_datafiltsector['count of whole Sector'] = Consumer_Services_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Distribution_Services_datafiltsector['count of whole Sector'] = Distribution_Services_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Electronic_Technology_datafiltsector['count of whole Sector'] = Electronic_Technology_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Health_Technology_datafiltsector['count of whole Sector'] = Health_Technology_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Producer_Manufacturing_datafiltsector['count of whole Sector'] = Producer_Manufacturing_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Retail_Trade_datafiltsector['count of whole Sector'] = Retail_Trade_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Technology_Services_datafiltsector['count of whole Sector'] = Technology_Services_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Transportation_datafiltsector['count of whole Sector'] = Transportation_datafiltsector.filter(regex='count of Stock').sum(axis=1)
Utilities_datafiltsector['count of whole Sector'] = Utilities_datafiltsector.filter(regex='count of Stock').sum(axis=1)

# add the mean of scores for all the stocks in specific sector
Commercial_services_datafiltsector['mean of whole Sector'] = Commercial_services_datafiltsector.filter(regex='Mean times').sum(axis=1)/Commercial_services_datafiltsector['count of whole Sector']
Communications_datafiltsector['mean of whole Sector'] = Communications_datafiltsector.filter(regex='Mean times').sum(axis=1)/Communications_datafiltsector['count of whole Sector']
Consumer_Durables_datafiltsector['mean of whole Sector'] = Consumer_Durables_datafiltsector.filter(regex='Mean times').sum(axis=1)/Consumer_Durables_datafiltsector['count of whole Sector']
Consumer_Non_Durables_datafiltsector['mean of whole Sector'] = Consumer_Non_Durables_datafiltsector.filter(regex='Mean times').sum(axis=1)/Consumer_Non_Durables_datafiltsector['count of whole Sector']
Consumer_Services_datafiltsector['mean of whole Sector'] = Consumer_Services_datafiltsector.filter(regex='Mean times').sum(axis=1)/Consumer_Services_datafiltsector['count of whole Sector']
Distribution_Services_datafiltsector['mean of whole Sector'] = Distribution_Services_datafiltsector.filter(regex='Mean times').sum(axis=1)/Distribution_Services_datafiltsector['count of whole Sector']
Electronic_Technology_datafiltsector['mean of whole Sector'] = Electronic_Technology_datafiltsector.filter(regex='Mean times').sum(axis=1)/Electronic_Technology_datafiltsector['count of whole Sector']
Health_Technology_datafiltsector['mean of whole Sector'] = Health_Technology_datafiltsector.filter(regex='Mean times').sum(axis=1)/Health_Technology_datafiltsector['count of whole Sector']
Producer_Manufacturing_datafiltsector['mean of whole Sector'] = Producer_Manufacturing_datafiltsector.filter(regex='Mean times').sum(axis=1)/Producer_Manufacturing_datafiltsector['count of whole Sector']
Retail_Trade_datafiltsector['mean of whole Sector'] = Retail_Trade_datafiltsector.filter(regex='Mean times').sum(axis=1)/Retail_Trade_datafiltsector['count of whole Sector']
Technology_Services_datafiltsector['mean of whole Sector'] = Technology_Services_datafiltsector.filter(regex='Mean times').sum(axis=1)/Technology_Services_datafiltsector['count of whole Sector']
Transportation_datafiltsector['mean of whole Sector'] = Transportation_datafiltsector.filter(regex='Mean times').sum(axis=1)/Transportation_datafiltsector['count of whole Sector']
Utilities_datafiltsector['mean of whole Sector'] = Utilities_datafiltsector.filter(regex='Mean times').sum(axis=1)/Utilities_datafiltsector['count of whole Sector']

# create dataframe to store sentiment scores mean and count by sector
datafiltsector_all = pd.DataFrame()

# add all the summary score values from each sector
datafiltsector_all['mean of Commercial services'] = Commercial_services_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Commercial services'] = Commercial_services_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Communications'] = Communications_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Communications'] = Communications_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Consumer Durables'] = Consumer_Durables_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Consumer Durables'] = Consumer_Durables_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Consumer Non Durables'] = Consumer_Non_Durables_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Consumer Non Durables'] = Consumer_Non_Durables_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Consumer Services'] = Consumer_Services_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Consumer Services'] = Consumer_Services_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Distribution Services'] = Distribution_Services_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Distribution Services'] = Distribution_Services_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Electronic Technology'] = Electronic_Technology_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Electronic Technology'] = Electronic_Technology_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Health Technology'] = Health_Technology_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Health Technology'] = Health_Technology_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Producer Manufacturing'] = Producer_Manufacturing_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Producer Manufacturing'] = Producer_Manufacturing_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Retail Trade'] = Retail_Trade_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Retail Trade'] = Retail_Trade_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Technology Services'] = Technology_Services_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Technology Services'] = Technology_Services_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Transportation'] = Transportation_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Transportation'] = Transportation_datafiltsector['count of whole Sector']
datafiltsector_all['mean of Utilities'] = Utilities_datafiltsector['mean of whole Sector']
datafiltsector_all['count of Utilities'] = Utilities_datafiltsector['count of whole Sector']

datafiltsector_all.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_filt_sectors_scores//All_filt_sectors_scores.csv')

