# imports needed for aggregating into sectors

import datetime as dt
import statistics
from calendar import week
from cmath import nan

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# section find the total market cap of all stocks
stock_list = ['AAPL','MSFT','AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','VRSN','SWKS','DOCU','SPLK','OKTA','CEG'
] # put CEG at the end since it has no market cap data ## might not include anywhere because of this

#create dataframe to store all market cap data
market_capdata = pd.DataFrame()

for i in stock_list:
    stock_name = str(i)
    # print(stock_name)

# import file containing comments
    data = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_Yahoo_Finance_Historical_Prices//'+str(stock_name)+'_Yahoo_Finance_Historical_Prices.csv')

    market_capdata['Market Cap of ' + str(stock_name)] = data['market cap']

market_capdata

pd.options.mode.chained_assignment = None

# loop to adjust market caps on same scale -> convert trillions values into billions
for i in market_capdata:
    stock_name = str(i)

    # extract last value to check if market cap of stock is in trillions or billions
    digit = market_capdata[str(stock_name)][0][-1]

    # remove the last string value form the market cap value
    market_capdata[str(stock_name)][0] = market_capdata[str(stock_name)][0][:-1]

    # transform string value into numeric
    market_capdata[str(stock_name)][0] = pd.to_numeric(market_capdata[str(stock_name)][0])

    if digit == 'T':
        # condition to multiply companies worth in the trillions to be mutliplied by 1000
        market_capdata[str(stock_name)] = market_capdata[str(stock_name)]*1000
    if digit == 'B':
        market_capdata[str(stock_name)] = market_capdata[str(stock_name)]

# remove all the other rows
market_capdata = market_capdata.iloc[:1]

# add column containing the total market cap
market_capdata['Total Market Cap'] = market_capdata.sum(axis=1)

# section about allocating stocks into respective sectors
# read all filtered stock data
datafilt = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

# read the file containing stocks with their respective sectors
stock_sectors = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Stock_sectors//Stock_sectors.csv')

# Create empty lists of each sector
# These sectors could be simplified if needed later
Commercial_Services = []
Communications = []
Consumer_Durables = []
Consumer_Non_Durables = []
Consumer_Services = []
Distribution_Services = []
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
        Commercial_Services.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Communications':
        Communications.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Durables':
        Consumer_Durables.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Non-Durables':
        Consumer_Non_Durables.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Consumer Services':
        Consumer_Services.append(stock_sectors['Ticker'][i])
    if stock_sectors['Sector'][i] == 'Distribution Services':
        Distribution_Services.append(stock_sectors['Ticker'][i])
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


# issue with XLNX appearing in data set
Electronic_Technology
Electronic_Technology.remove('XLNX')
Electronic_Technology

# issue with PTON appearing in data set
Consumer_Services
Consumer_Services.remove('PTON')
Consumer_Services


# create new dataframes containing the scores for each sector
Commercial_Services_datafiltsector = pd.DataFrame()
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

# seperate the stocks into their respective sectors to facilitate calculations in the next step
for i in datafilt:
    # print(i)

    Commercial_Services_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Commercial_Services:
        if str(i) == 'weekly return of Stock ' + j:
            Commercial_Services_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Commercial_Services_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Commercial_Services_datafiltsector['Total of Commercial Services'] = Commercial_Services_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Communications_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Communications:
        if str(i) == 'weekly return of Stock ' + j:
            Communications_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Communications_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Communications_datafiltsector['Total of Communications'] = Communications_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Consumer_Durables_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Consumer_Durables:
        if str(i) == 'weekly return of Stock ' + j:
            Consumer_Durables_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Consumer_Durables_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Consumer_Durables_datafiltsector['Total of Consumer Durables'] = Consumer_Durables_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Consumer_Non_Durables_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Consumer_Non_Durables:
        if str(i) == 'weekly return of Stock ' + j:
            Consumer_Non_Durables_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Consumer_Non_Durables_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j] 
    
    Consumer_Non_Durables_datafiltsector['Total of Consumer Non Durables'] = Consumer_Non_Durables_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Consumer_Services_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Consumer_Services:
        if str(i) == 'weekly return of Stock ' + j:
            Consumer_Services_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Consumer_Services_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]     

    Consumer_Services_datafiltsector['Total of Consumer Services'] = Consumer_Services_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Distribution_Services_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Distribution_Services:
        if str(i) == 'weekly return of Stock ' + j:
            Distribution_Services_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Distribution_Services_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j] 

    Distribution_Services_datafiltsector['Total of Distribution Services'] =  Distribution_Services_datafiltsector.filter(regex='Market Cap').sum(axis=1)



    Electronic_Technology_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']
            
    for j in Electronic_Technology:
        if str(i) == 'weekly return of Stock ' + j:
            Electronic_Technology_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Electronic_Technology_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Electronic_Technology_datafiltsector['Total of Electronic Technology'] =  Electronic_Technology_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Health_Technology_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Health_Technology:
        if str(i) == 'weekly return of Stock ' + j:
            Health_Technology_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Health_Technology_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Health_Technology_datafiltsector['Total of Health Technology'] =  Health_Technology_datafiltsector.filter(regex='Market Cap').sum(axis=1)



    Producer_Manufacturing_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Producer_Manufacturing:
        if str(i) == 'weekly return of Stock ' + j:
            Producer_Manufacturing_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Producer_Manufacturing_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Producer_Manufacturing_datafiltsector['Total of Producer Manufacturing'] =  Producer_Manufacturing_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Retail_Trade_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Retail_Trade:
        if str(i) == 'weekly return of Stock ' + j:
            Retail_Trade_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Retail_Trade_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Retail_Trade_datafiltsector['Total of Retail Trade'] =  Retail_Trade_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Technology_Services_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Technology_Services:
        if str(i) == 'weekly return of Stock ' + j:
            Technology_Services_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Technology_Services_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Technology_Services_datafiltsector['Total of Technology Services'] =  Technology_Services_datafiltsector.filter(regex='Market Cap').sum(axis=1)


    Transportation_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Transportation:
        if str(i) == 'weekly return of Stock ' + j:
            Transportation_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Transportation_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Transportation_datafiltsector['Total of Transportation'] =  Transportation_datafiltsector.filter(regex='Market Cap').sum(axis=1)



    Utilities_datafiltsector['weekly dates'] = datafilt['weekly date of Stock']

    for j in Utilities:
        if str(i) == 'weekly return of Stock ' + j:
            Utilities_datafiltsector['weekly return of Stock ' + j] = datafilt['weekly return of Stock ' + j]
            Utilities_datafiltsector['Market Cap of Stock ' + j] = market_capdata['Market Cap of '+j]

    Utilities_datafiltsector['Total of Utilities'] =  Utilities_datafiltsector.filter(regex='Market Cap').sum(axis=1)


# Sectors_list = [Commercial_services, Communications,Consumer_Durables ,Consumer_Non_Durables ,Consumer_Services ,Distribution_Servcies ,Electronic_Technology ,Health_Technology ,Producer_Manufacturing ,Retail_Trade ,Technology_Services,Transportation,Utilities]
# Alldatafiltsector = [Commercial_services_datafiltsector,Communications_datafiltsector,Consumer_Durables_datafiltsector,Consumer_Non_Durables_datafiltsector,Consumer_Services_datafiltsector,Distribution_Services_datafiltsector,Electronic_Technology_datafiltsector,Health_Technology_datafiltsector,Producer_Manufacturing_datafiltsector,Retail_Trade_datafiltsector,Technology_Services_datafiltsector,Transportation_datafiltsector,Utilities_datafiltsector]

for j in Commercial_Services:
    Commercial_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Commercial_Services_datafiltsector.index:
    # include market cap if the stock is active
    for j in Commercial_Services:
        if pd.notnull(Commercial_Services_datafiltsector['weekly return of Stock ' + j][k]):
            Commercial_Services_datafiltsector['Market Cap of Stock '+j][k] = Commercial_Services_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Commercial_Services_datafiltsector['Total of Commercial Services'][k] = Commercial_Services_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Commercial_Services:
       Commercial_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Commercial_Services_datafiltsector['Market Cap of Stock '+j][k]/Commercial_Services_datafiltsector['Total of Commercial Services'][k]

    # Sector portfolio return adjustment
       Commercial_Services_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Commercial_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Commercial_Services_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Commercial_Services_datafiltsector['Sector market cap weighted weekly return'] = Commercial_Services_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1



for j in Communications:
    Communications_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Communications_datafiltsector.index:
    # include market cap if the stock is active
    for j in Communications:
        if pd.notnull(Communications_datafiltsector['weekly return of Stock ' + j][k]):
            Communications_datafiltsector['Market Cap of Stock '+j][k] = Communications_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Communications_datafiltsector['Total of Communications'][k] = Communications_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Communications:
       Communications_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Communications_datafiltsector['Market Cap of Stock '+j][k]/Communications_datafiltsector['Total of Communications'][k]

    # Sector portfolio return adjustment
       Communications_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Communications_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Communications_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Communications_datafiltsector['Sector market cap weighted weekly return'] = Communications_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1


for j in Consumer_Durables:
    Consumer_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Consumer_Durables_datafiltsector.index:
    # include market cap if the stock is active
    for j in Consumer_Durables:
        if pd.notnull(Consumer_Durables_datafiltsector['weekly return of Stock ' + j][k]):
            Consumer_Durables_datafiltsector['Market Cap of Stock '+j][k] = Consumer_Durables_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Consumer_Durables_datafiltsector['Total of Consumer Durables'][k] = Consumer_Durables_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Consumer_Durables:
       Consumer_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Consumer_Durables_datafiltsector['Market Cap of Stock '+j][k]/Consumer_Durables_datafiltsector['Total of Consumer Durables'][k]

    # Sector portfolio return adjustment
       Consumer_Durables_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Consumer_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Consumer_Durables_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Consumer_Durables_datafiltsector['Sector market cap weighted weekly return'] = Consumer_Durables_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1

Consumer_Durables_datafiltsector.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testinggg.csv')

for j in Consumer_Non_Durables:
    Consumer_Non_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Consumer_Non_Durables_datafiltsector.index:
    # include market cap if the stock is active
    for j in Consumer_Non_Durables:
        if pd.notnull(Consumer_Non_Durables_datafiltsector['weekly return of Stock ' + j][k]):
            Consumer_Non_Durables_datafiltsector['Market Cap of Stock '+j][k] = Consumer_Non_Durables_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Consumer_Non_Durables_datafiltsector['Total of Consumer Non Durables'][k] = Consumer_Non_Durables_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Consumer_Non_Durables:
       Consumer_Non_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Consumer_Non_Durables_datafiltsector['Market Cap of Stock '+j][k]/Consumer_Non_Durables_datafiltsector['Total of Consumer Non Durables'][k]

    # Sector portfolio return adjustment
       Consumer_Non_Durables_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Consumer_Non_Durables_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Consumer_Non_Durables_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Consumer_Non_Durables_datafiltsector['Sector market cap weighted weekly return'] = Consumer_Non_Durables_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1



for j in Consumer_Services:
    Consumer_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Consumer_Services_datafiltsector.index:
    # include market cap if the stock is active
    for j in Consumer_Services:
        if pd.notnull(Consumer_Services_datafiltsector['weekly return of Stock ' + j][k]):
            Consumer_Services_datafiltsector['Market Cap of Stock '+j][k] = Consumer_Services_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Consumer_Services_datafiltsector['Total of Consumer Services'][k] = Consumer_Services_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Consumer_Services:
       Consumer_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Consumer_Services_datafiltsector['Market Cap of Stock '+j][k]/Consumer_Services_datafiltsector['Total of Consumer Services'][k]

    # Sector portfolio return adjustment
       Consumer_Services_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Consumer_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Consumer_Services_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Consumer_Services_datafiltsector['Sector market cap weighted weekly return'] = Consumer_Services_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1




for j in Distribution_Services:
    Distribution_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Distribution_Services_datafiltsector.index:
    # include market cap if the stock is active
    for j in Distribution_Services:
        if pd.notnull(Distribution_Services_datafiltsector['weekly return of Stock ' + j][k]):
            Distribution_Services_datafiltsector['Market Cap of Stock '+j][k] = Distribution_Services_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Distribution_Services_datafiltsector['Total of Distribution Services'][k] = Distribution_Services_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Distribution_Services:
       Distribution_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Distribution_Services_datafiltsector['Market Cap of Stock '+j][k]/Distribution_Services_datafiltsector['Total of Distribution Services'][k]

    # Sector portfolio return adjustment
       Distribution_Services_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Distribution_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Distribution_Services_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Distribution_Services_datafiltsector['Sector market cap weighted weekly return'] = Distribution_Services_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1



for j in Electronic_Technology:
    Electronic_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Electronic_Technology_datafiltsector.index:
    # include market cap if the stock is active
    for j in Electronic_Technology:
        if pd.notnull(Electronic_Technology_datafiltsector['weekly return of Stock ' + j][k]):
            Electronic_Technology_datafiltsector['Market Cap of Stock '+j][k] = Electronic_Technology_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Electronic_Technology_datafiltsector['Total of Electronic Technology'][k] = Electronic_Technology_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Electronic_Technology:
       Electronic_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Electronic_Technology_datafiltsector['Market Cap of Stock '+j][k]/Electronic_Technology_datafiltsector['Total of Electronic Technology'][k]

    # Sector portfolio return adjustment
       Electronic_Technology_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Electronic_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Electronic_Technology_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Electronic_Technology_datafiltsector['Sector market cap weighted weekly return'] = Electronic_Technology_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1


for j in Health_Technology:
    Health_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Health_Technology_datafiltsector.index:
    # include market cap if the stock is active
    for j in Health_Technology:
        if pd.notnull(Health_Technology_datafiltsector['weekly return of Stock ' + j][k]):
            Health_Technology_datafiltsector['Market Cap of Stock '+j][k] = Health_Technology_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Health_Technology_datafiltsector['Total of Health Technology'][k] = Health_Technology_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Health_Technology:
       Health_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Health_Technology_datafiltsector['Market Cap of Stock '+j][k]/Health_Technology_datafiltsector['Total of Health Technology'][k]

    # Sector portfolio return adjustment
       Health_Technology_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Health_Technology_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Health_Technology_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Health_Technology_datafiltsector['Sector market cap weighted weekly return'] = Health_Technology_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1



for j in Producer_Manufacturing:
    Producer_Manufacturing_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Producer_Manufacturing_datafiltsector.index:
    # include market cap if the stock is active
    for j in Producer_Manufacturing:
        if pd.notnull(Producer_Manufacturing_datafiltsector['weekly return of Stock ' + j][k]):
            Producer_Manufacturing_datafiltsector['Market Cap of Stock '+j][k] = Producer_Manufacturing_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Producer_Manufacturing_datafiltsector['Total of Producer Manufacturing'][k] = Producer_Manufacturing_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Producer_Manufacturing:
       Producer_Manufacturing_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Producer_Manufacturing_datafiltsector['Market Cap of Stock '+j][k]/Producer_Manufacturing_datafiltsector['Total of Producer Manufacturing'][k]

    # Sector portfolio return adjustment
       Producer_Manufacturing_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Producer_Manufacturing_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Producer_Manufacturing_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Producer_Manufacturing_datafiltsector['Sector market cap weighted weekly return'] = Producer_Manufacturing_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1


for j in Retail_Trade:
    Retail_Trade_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Retail_Trade_datafiltsector.index:
    # include market cap if the stock is active
    for j in Retail_Trade:
        if pd.notnull(Retail_Trade_datafiltsector['weekly return of Stock ' + j][k]):
            Retail_Trade_datafiltsector['Market Cap of Stock '+j][k] = Retail_Trade_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Retail_Trade_datafiltsector['Total of Retail Trade'][k] = Retail_Trade_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Retail_Trade:
       Retail_Trade_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Retail_Trade_datafiltsector['Market Cap of Stock '+j][k]/Retail_Trade_datafiltsector['Total of Retail Trade'][k]

    # Sector portfolio return adjustment
       Retail_Trade_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Retail_Trade_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Retail_Trade_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Retail_Trade_datafiltsector['Sector market cap weighted weekly return'] = Retail_Trade_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1


for j in Technology_Services:
    Technology_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Technology_Services_datafiltsector.index:
    # include market cap if the stock is active
    for j in Technology_Services:
        if pd.notnull(Technology_Services_datafiltsector['weekly return of Stock ' + j][k]):
            Technology_Services_datafiltsector['Market Cap of Stock '+j][k] = Technology_Services_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Technology_Services_datafiltsector['Total of Technology Services'][k] = Technology_Services_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Technology_Services:
       Technology_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Technology_Services_datafiltsector['Market Cap of Stock '+j][k]/Technology_Services_datafiltsector['Total of Technology Services'][k]

    # Sector portfolio return adjustment
       Technology_Services_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Technology_Services_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Technology_Services_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Technology_Services_datafiltsector['Sector market cap weighted weekly return'] = Technology_Services_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1


for j in Transportation:
    Transportation_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Transportation_datafiltsector.index:
    # include market cap if the stock is active
    for j in Transportation:
        if pd.notnull(Transportation_datafiltsector['weekly return of Stock ' + j][k]):
            Transportation_datafiltsector['Market Cap of Stock '+j][k] = Transportation_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Transportation_datafiltsector['Total of Transportation'][k] =Transportation_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Transportation:
       Transportation_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Transportation_datafiltsector['Market Cap of Stock '+j][k]/Transportation_datafiltsector['Total of Transportation'][k]

    # Sector portfolio return adjustment
       Transportation_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Transportation_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Transportation_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Transportation_datafiltsector['Sector market cap weighted weekly return'] = Transportation_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1



    # new loop to perform the rest of the calculations
for j in Utilities:
    Utilities_datafiltsector['Percentage of Market Cap Sector for Stock ' + j] = np.nan

for k in Utilities_datafiltsector.index:
    # include market cap if the stock is active
    for j in Utilities:
        if pd.notnull(Utilities_datafiltsector['weekly return of Stock ' + j][k]):
            Utilities_datafiltsector['Market Cap of Stock '+j][k] = Utilities_datafiltsector['Market Cap of Stock '+j][0]

    # recalculate the market cap for the specific time frame
    Utilities_datafiltsector['Total of Utilities'][k] = Utilities_datafiltsector.loc[k].filter(regex='Market Cap').sum(axis=0)

    # calculate the market cap percentage of Sector for every stock
    for j in Utilities:
       Utilities_datafiltsector['Percentage of Market Cap Sector for Stock ' + j][k] = Utilities_datafiltsector['Market Cap of Stock '+j][k]/Utilities_datafiltsector['Total of Utilities'][k]

    # Sector portfolio return adjustment
       Utilities_datafiltsector['Adjusted sector weighted weekly return for Stock ' + j] = Utilities_datafiltsector['Percentage of Market Cap Sector for Stock ' + j]*(Utilities_datafiltsector['weekly return of Stock ' + j]-1)
    
    # Sector market cap weighted weekly return
       Utilities_datafiltsector['Sector market cap weighted weekly return'] = Utilities_datafiltsector.filter(regex='Adjusted').sum(axis=1)+1

# section to put all sector market weighted filtered returns together
All_sectors_adjusted_returns = pd.DataFrame()



# add the dates of the returns
All_sectors_adjusted_returns['weekly dates'] = datafilt['weekly date of Stock']

# add the sector stock weighted returns
All_sectors_adjusted_returns['Commercial Services returns'] = Commercial_Services_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Communications returns'] = Communications_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Consumer Durables returns'] = Consumer_Durables_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Consumer Non Durables returns'] = Consumer_Non_Durables_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Consumer Services returns'] = Consumer_Services_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Distribution Services returns'] = Distribution_Services_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Electronic Technology returns'] = Electronic_Technology_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Health Technology returns'] = Health_Technology_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Producer Manufacturing returns'] = Producer_Manufacturing_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Retail Trade returns'] = Retail_Trade_datafiltsector['Sector market cap weighted weekly return'] 
All_sectors_adjusted_returns['Technology Services returns'] = Technology_Services_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Transportation returns'] = Transportation_datafiltsector['Sector market cap weighted weekly return']
All_sectors_adjusted_returns['Utilities returns'] = Utilities_datafiltsector['Sector market cap weighted weekly return']

# add the market caps of each sector
All_sectors_adjusted_returns['Commercial Services Market Cap'] = Commercial_Services_datafiltsector['Total of Commercial Services']
All_sectors_adjusted_returns['Communications Market Cap'] = Communications_datafiltsector['Total of Communications']
All_sectors_adjusted_returns['Consumer Durables Market Cap'] = Consumer_Durables_datafiltsector['Total of Consumer Durables']
All_sectors_adjusted_returns['Consumer Non Durables Market Cap'] = Consumer_Non_Durables_datafiltsector['Total of Consumer Non Durables']
All_sectors_adjusted_returns['Consumer Services Market Cap'] = Consumer_Services_datafiltsector['Total of Consumer Services']
All_sectors_adjusted_returns['Distribution Services Market Cap'] = Distribution_Services_datafiltsector['Total of Distribution Services']
All_sectors_adjusted_returns['Electronic Technology Market Cap'] = Electronic_Technology_datafiltsector['Total of Electronic Technology']
All_sectors_adjusted_returns['Health Technology Market Cap'] = Health_Technology_datafiltsector['Total of Health Technology']
All_sectors_adjusted_returns['Producer Manufacturing Market Cap'] = Producer_Manufacturing_datafiltsector['Total of Producer Manufacturing']
All_sectors_adjusted_returns['Retail Trade Market Cap'] = Retail_Trade_datafiltsector['Total of Retail Trade'] 
All_sectors_adjusted_returns['Technology Services Market Cap'] = Technology_Services_datafiltsector['Total of Technology Services']
All_sectors_adjusted_returns['Transportation Market Cap'] = Transportation_datafiltsector['Total of Transportation']
All_sectors_adjusted_returns['Utilities Market Cap'] = Utilities_datafiltsector['Total of Utilities']


# save the sector weekly returns 
All_sectors_adjusted_returns.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_StockweightedSectors_Returns//All_StockweightedSectors_Returns.csv')



## section to agglomerate returns into quarterly returns

# list of sectors
sector_list = ['Commercial Services','Communications','Consumer Durables','Consumer Non Durables','Consumer Services','Distribution Services','Electronic Technology'
,'Health Technology','Producer Manufacturing','Retail Trade','Technology Services','Transportation','Utilities']

#create new dataframe to store quarterly scores
dataretquart = pd.DataFrame()


#create dataframe to perform interim manipulations
dataretmanip =pd.DataFrame()

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
    mask = (All_sectors_adjusted_returns['weekly dates'] > start_date) & (All_sectors_adjusted_returns['weekly dates'] <= end_date)
    dataretmanip = All_sectors_adjusted_returns.loc[mask]

    dataretmanip

    # loop each column to average the weekly scores
    for i in sector_list:
        sector_name = str(i)
        # print(sector_name)

        # calculate the mean of average weekly scores for each sector 
        dataretmanip[str(sector_name)+' returns'] = dataretmanip[str(sector_name)+' returns'].product()
        # get the market cap sector for each quarter, median is used since the first time quarter has a different value
        dataretmanip[str(sector_name)+' Market Cap'] = dataretmanip[str(sector_name)+' Market Cap'].median()

    dataretmanip['Total Market Cap'] = dataretmanip.filter(regex='Market Cap').sum(axis=1)

    # loop to obtain percentage of total market cap
    for i in sector_list:
        sector_name = str(i)
        # print(sector_name)

        # calculate the mean of average weekly scores for each sector 
        dataretmanip[str(sector_name)+' percentage of Total Market Cap'] = dataretmanip[str(sector_name)+' Market Cap']/dataretmanip['Total Market Cap']


    dataretmanip
    # only keep the first row of data, since they all contain the same values and makes it easier for appending with the other data points
    dataretmanip = dataretmanip.loc[dataretmanip.index[0]]

    dataretmanip

    # identify each quarter with the starting date of the quarter
    dataretmanip['weekly dates'] = start_date


    dataretquart = dataretquart.append(dataretmanip)

dataretquart 

# save the quarterly scores
dataretquart.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Quarterly_All_StockweightedSectors_Returns//Quarterly_All_StockweightedSectors_Returns.csv')
   









