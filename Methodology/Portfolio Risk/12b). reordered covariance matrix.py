# imports needed
from calendar import week
from cmath import nan, sqrt
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp
import statsmodels.api as sm
import seaborn as sns

# get relevant dataframes
dataewmasentfactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Sentiment_factor//Sentiment_factor.csv')

datafamafactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Fama_French_Factors//F-F_Research_Data_5_Factors_2x3.csv')

datafiltweek = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

regression_values = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Regression//All_Stocks_Regression.csv')

labels_order_minussentiment = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//labels_order//labels_order_minussentiment.csv')

labels_order = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//labels_order//labels_order.csv')

# list of stocks
stock_list = ['AAPL','MSFT' ,'AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','VRSN','SWKS','DOCU','SPLK','OKTA','CEG']

# reorder the stocks based on the clustering found previously
stock_list_clustered = []

labels_order

counter = 0

for i in labels_order['0']:
    print(counter)
    print(i)
    stock_list[i]
    stock_list_clustered.insert(counter,stock_list[i])
    counter = counter+1 

stock_list_clustered

# reorder the stocks based on the clustering found previously without sentiment

stock_list_clustered_minussentiment = []

labels_order_minussentiment

counter = 0

for i in labels_order_minussentiment['0']:
    print(counter)
    print(i)
    stock_list[i]
    stock_list_clustered_minussentiment.insert(counter,stock_list[i])
    counter = counter+1 

stock_list_clustered_minussentiment
stock_list_clustered


# merge both dataframes together

dataallfactors = datafamafactor.copy()

dataallfactors['Sentiment factor'] = dataewmasentfactor['weekly sentiment factor']

datafiltweek = datafiltweek[:-3]
dataallfactors = dataallfactors[:-3]

# add column name to weekly dates
dataallfactors.columns = ['weekly dates','Mkt-RF', 'SMB', 'HML','RF','Sentiment factor']


# remove rows where sentiment factor doesn't exist yet or the result is flawed
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



# create new dataframe to include all factors except the risk free rate
dataallfactorswithoutRF = pd.DataFrame()

#modify dataframe to only get relevant colunmns
dataallfactorswithoutRF = dataallfactors.drop(['RF','weekly dates'], axis=1) 

dataallfactorswithoutRF = dataallfactorswithoutRF.transpose()

dataallfactorswithoutRF

# perform covariance of factors
covariance_factors = np.cov(dataallfactorswithoutRF)

#covariance_factors = covariance_factors.astype(int)

covariance_factors = pd.DataFrame(data=covariance_factors)

# modify regression values of stocks to only keep relevant values
regression_values = regression_values.drop(0, axis=0) 
regression_values = regression_values.drop(['Unnamed: 0'], axis=1) 

# reset columns and rows index
regression_values.index, regression_values.columns = [range(regression_values.index.size), range(regression_values.columns.size)]

regression_values = regression_values.astype(float)

regression_values_t = regression_values.transpose()

# reorder regression values

regression_values_clustered_t = pd.DataFrame()

counter = 0

# loop to reassign values order
for i in labels_order['0']:
    regression_values_clustered_t[counter] = regression_values_t[i]
    counter = counter+1

regression_values_clustered_t

regression_values_t = regression_values_clustered_t

regression_values = regression_values_t.transpose()

regression_values

# perform matrix calculations

# Create interim matrices for the calculation
Formula = pd.DataFrame()
Formula = regression_values.copy()
zero_data = np.zeros (shape= (102,102))
Results = pd.DataFrame(zero_data)

# empty dataframe for matrix calculation
for col in Formula.columns:
    Formula[col].values[:] = 0

# first part of the matrix calculation
for i in range(len(regression_values)):

    for j in range(len(covariance_factors.columns)):

        for k in range(len(covariance_factors)):
            Formula.loc[i,j] += regression_values.loc[i,k] * covariance_factors.loc[k,j]

Formula.shape

# second part of the matrix calculation
for i in range(len(Formula)):

    for j in range(len(regression_values_t.columns)):

        for k in range(len(regression_values_t)):
            Results.loc[i,j] += Formula.loc[i,k] * regression_values_t.loc[k,j]

Results.shape

## section to calculate residual variance of each stock

residual_timeseries = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Residual_time_series//All_Stocks_Residual_time_series.csv')

residual_timeseries

# loop to calculate the variance of each stock
for i in stock_list_clustered:
    stock_name = str(i)
    residual_timeseries[str(stock_name)+' residuals variance'] = residual_timeseries['Residual time series of Stock '+str(stock_name)].var()

residual_timeseries

## section to create a matrix containing all the residual variances

zero_data = np.zeros (shape= (102,102))
Residualvariance_diagonal = pd.DataFrame(zero_data)

counter = 0 

# loop to add residual variance to each line and column of the matrix
for i in stock_list_clustered:
    stock_name = str(i)

    Residualvariance_diagonal.loc[counter,counter] = residual_timeseries.loc[1,str(stock_name)+' residuals variance']

    counter = counter+1



## section to add the residual variance to the covariance matrix

# add both matrices together
Results = Results + Residualvariance_diagonal

Results = Results*52

Results





### section to calculate covariance without the sentiment factor

# create new covariance without the sentiment factor

# create new dataframe to store the factors
dataallfactorswithoutsentiment = pd.DataFrame()

dataallfactorswithoutsentiment = dataallfactors.drop(['Sentiment factor'], axis=1)


#modify dataframe to only get relevant colunmns
dataallfactorswithoutsentiment_RF = dataallfactorswithoutsentiment.drop(['RF','weekly dates'], axis=1) 

dataallfactorswithoutsentiment_RF = dataallfactorswithoutsentiment_RF.transpose()

dataallfactorswithoutsentiment_RF

# perform covariance of factors
covariance_factors = np.cov(dataallfactorswithoutsentiment_RF)

#covariance_factors = covariance_factors.astype(int)

covariance_factors = pd.DataFrame(data=covariance_factors)

# modify regression values of stocks to only keep relevant values

# reset regression values dataframe
regression_values = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Regression//All_Stocks_Regression_minussentiment.csv')

regression_values

regression_values = regression_values.drop(0, axis=0) 
regression_values = regression_values.drop(['Unnamed: 0'], axis=1) 

# reset columns and rows index
regression_values.index, regression_values.columns = [range(regression_values.index.size), range(regression_values.columns.size)]

regression_values = regression_values.astype(float)

regression_values_t = regression_values.transpose()

# reorder regression values

regression_values_clustered_t = pd.DataFrame()

counter = 0

# loop to reassign values order
for i in labels_order['0']:
    regression_values_clustered_t[counter] = regression_values_t[i]
    counter = counter+1

regression_values_clustered_t

regression_values_t = regression_values_clustered_t

regression_values = regression_values_t.transpose()

regression_values

# perform matrix calculations

# Create interim matrices for the calculation
Formula = pd.DataFrame()
Formula = regression_values.copy()
zero_data = np.zeros (shape= (102,102))
Results1 = pd.DataFrame(zero_data)

# empty dataframe for matrix calculation
for col in Formula.columns:
    Formula[col].values[:] = 0


# regression_values@covariance_factors is a faster way

# first part of the matrix calculation
for i in range(len(regression_values)):

    for j in range(len(covariance_factors.columns)):

        for k in range(len(covariance_factors)):
            Formula.loc[i,j] += regression_values.loc[i,k] * covariance_factors.loc[k,j]

Formula

Formula.shape

# second part of the matrix calculation
for i in range(len(Formula)):

    for j in range(len(regression_values_t.columns)):

        for k in range(len(regression_values_t)):
            Results1.loc[i,j] += Formula.loc[i,k] * regression_values_t.loc[k,j]

Results1.shape

Results1

## section to calculate residual variance of each stock

residual_timeseries = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Stocks_Residual_time_series//All_Stocks_Residual_time_series_minussentiment.csv')

residual_timeseries

# loop to calculate the variance of each stock
for i in stock_list_clustered:
    stock_name = str(i)
    residual_timeseries[str(stock_name)+' residuals variance'] = residual_timeseries['Residual time series of Stock '+str(stock_name)].var()

residual_timeseries

## section to create a matrix containing all the residual variances

zero_data = np.zeros (shape= (102,102))
Residualvariance_diagonal = pd.DataFrame(zero_data)

counter = 0 

# loop to add residual variance to each line and column of the matrix
for i in stock_list_clustered:
    stock_name = str(i)

    Residualvariance_diagonal.loc[counter,counter] = residual_timeseries.loc[1,str(stock_name)+' residuals variance']

    counter = counter+1



## section to add the residual variance to the covariance matrix

# add both matrices together
Results1 = Results1 + Residualvariance_diagonal

# annualise
Results1 = Results1*52

Results
Results1

# save both covariance matrices
Results.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Covariance_Matrix_Clustered//Covariance_Matrix_Clustered.csv')
Results1.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Covariance_Matrix_Clustered//Covariance_Matrix_minussentiment_Clustered.csv')

## section to show clustered correlations

#bhierarchical cluestering
covmat = Results
covmat_minussentiment = Results1

## section to show the covariance matrix clustering
covmat.shape

corrmat = np.corrcoef(covmat,rowvar=False)

corrmat

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

mp.figure(figsize=(15,10))

#sns.heatmap(np.round(corrmat,2), cmap='RdBu', annot=True, 
#            annot_kws={"size": 7}, vmin=-1, vmax=1)

#mp.show()

covmat_minussentiment.shape

corrmat_minussentiment = np.corrcoef(covmat_minussentiment,rowvar=False)

corrmat_minussentiment

mp.figure(figsize=(15,10))

#sns.heatmap(np.round(corrmat_minussentiment,2), cmap='RdBu', annot=True, 
#            annot_kws={"size": 7}, vmin=-1, vmax=1)

#mp.show()

## heatmap of the clustered correlation
hm = sns.heatmap(corrmat,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)

mp.show()

hm = sns.heatmap(corrmat_minussentiment,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)

mp.show()

hm = sns.heatmap(corrmat-corrmat_minussentiment,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)
mp.title('Correlation matrix indicating the difference between correlation matrix with and without sentiment')
mp.show()


mp.figure(figsize=(15,10))

sns.heatmap(np.round(corrmat-corrmat_minussentiment,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1)

mp.show()


sns.clustermap(corrmat-corrmat_minussentiment, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12));

mp.show()



### section to calculate portfolio volatility assuming equally weights
# create equally weighted stock weights
ones_data = np.ones (shape= (102,1))
equal_weights = pd.DataFrame(ones_data)
equal_weights = equal_weights/len(equal_weights)

equal_weights = equal_weights.transpose()

equal_weights_t = equal_weights.transpose()



# create interim formula matrix
Formula = pd.DataFrame()
Formula = equal_weights.copy()
# empty dataframe for matrix calculation
for col in Formula.index:
    Formula[col].values[:] = 0

Variance_portfolio = np.zeros (shape= (1,1))
Variance_portfolio = pd.DataFrame(data = Variance_portfolio)

# first part of the matrix calculation
for i in range(len(equal_weights)):

    for j in range(len(Results.columns)):

        for k in range(len(Results)):
            Formula.loc[i,j] += equal_weights.loc[i,k] * Results.loc[k,j]

Formula.shape


# second part of the matrix calculation
for i in range(len(Formula)):

    for j in range(len(equal_weights_t.columns)):

        for k in range(len(equal_weights_t)):
            Variance_portfolio.loc[i,j] += Formula.loc[i,k] * equal_weights_t.loc[k,j]

Variance_portfolio.shape

Variance_portfolio = Variance_portfolio**0.5

Variance_portfolio

### section to calculate portfolio volatility assuming equally weights with the sentiment factor

# create equally weighted stock weights
ones_data = np.ones (shape= (102,1))
equal_weights = pd.DataFrame(ones_data)
equal_weights = equal_weights/len(equal_weights)

equal_weights = equal_weights.transpose()

equal_weights_t = equal_weights.transpose()



# create interim formula matrix
Formula = pd.DataFrame()
Formula = equal_weights.copy()
# empty dataframe for matrix calculation
for col in Formula.index:
    Formula[col].values[:] = 0

Variance_portfolio_minussentiment = np.zeros (shape= (1,1))
Variance_portfolio_minussentiment = pd.DataFrame(data = Variance_portfolio_minussentiment)


# first part of the matrix calculation
for i in range(len(equal_weights)):

    for j in range(len(Results1.columns)):

        for k in range(len(Results1)):
            Formula.loc[i,j] += equal_weights.loc[i,k] * Results1.loc[k,j]

Formula.shape


# second part of the matrix calculation
for i in range(len(Formula)):

    for j in range(len(equal_weights_t.columns)):

        for k in range(len(equal_weights_t)):
            Variance_portfolio_minussentiment.loc[i,j] += Formula.loc[i,k] * equal_weights_t.loc[k,j]

Variance_portfolio_minussentiment.shape

# obtain the volatility of portfolio by performing the square root
Variance_portfolio_minussentiment = Variance_portfolio_minussentiment**0.5

Variance_portfolio
Variance_portfolio_minussentiment

