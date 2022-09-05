# imports needed
from calendar import week
from cmath import nan
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as plt
import statsmodels.api as sm

# get relevant dataframes
dataewmasentfactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_summarised_normalised_VADER_scores//All_summarised_Topic_normalised_VADER_scores.csv')

datafamafactor = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Fama_French_Factors//F-F_Research_Data_5_Factors_2x3_daily.csv')

datafiltday = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Topic_Yahoo_Historical_prices//All_stock_Topic_Yahoo_Historical_prices.csv')

datafamafactor

datafiltday

dataewmasentfactor

# list of stocks
stock_list = ["GME","AMC","BB","NOK","BBBY"]
# add the sentiment factor 

dataallfactors = datafamafactor.copy()

# create column to contain the sentiment factor
for i in stock_list:
    stock_name = str(i)
    dataallfactors['Sentiment factor of Stock '+str(stock_name)] = dataallfactors['RF']
    dataallfactors['Sentiment factor of Stock '+str(stock_name)] = 0

dataallfactors
# set the first value of the sentiment factor
# dataallfactors.loc[0,'Unnamed: 0'] = dataewmasentfactor.loc[0,'weekly dates']
# add a counter to make loop process faster
# loop to add the values of sentiment to the factor list
for g in stock_list:
    # add a counter to make loop process faster
    counter = 2300
    print(g)
    stock_name = str(g)
    for i in range(counter,dataallfactors.shape[0]):
        print(i)
        for j in range(counter,dataewmasentfactor.shape[0]): 
            if dataallfactors.loc[i,'Unnamed: 0'] == dataewmasentfactor.loc[j,'daily date']:
                dataallfactors.loc[i,'Sentiment factor of Stock '+str(stock_name)] = dataewmasentfactor.loc[j,'mean of Stock '+str(stock_name)]
                counter = counter + 1
                continue



dataallfactors.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testing.csv')


# fill na values with 0

dataallfactors = dataallfactors.fillna(0)

dataallfactors

# merge both dataframes together

datafiltday = datafiltday[:-23]
dataallfactors = dataallfactors

datafiltday
dataallfactors

# remove rows where sentiment factor doesn't exist yet
#for i in dataallfactors.index:

    #if dataallfactors.loc[i,'Sentiment factor'] == 0:

    #    dataallfactors = dataallfactors.drop(i)
    #    datafiltday = datafiltday.drop(i)
    #    continue
        
    #if dataallfactors.loc[i,'Sentiment factor'] >= 90:

    #    dataallfactors = dataallfactors.drop(i)
    #    datafiltday = datafiltday.drop(i)
    #    continue

    #if dataallfactors.loc[i,'Sentiment factor'] <= -90:

    #    dataallfactors = dataallfactors.drop(i)
    #    datafiltday = datafiltday.drop(i)
    #   continue

## section where regression is performed

# create dataframe to include all regression betas for each stock
regression_values = pd.DataFrame()
regression_values['regression parameters'] = 'b0','b1','b2','b3','b4','b5','b6'

regression_values

# fill NA values in the return data
datafiltday = datafiltday.fillna(value=1)

datafiltday

# add column name to weekly dates
dataallfactors.columns = ['daily dates','Mkt-RF', 'SMB', 'HML','RMW','CMA','RF','Sentiment factor']

dataallfactors

for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # create copies of dataframes to be modified when certain stock returns aren't available
    datafiltdaymod = datafiltday.copy()
    dataallfactorsmod = dataallfactors.copy()

    # loop to remove rows where stock return doesn't exist yet
    for j in datafiltday.index:

        if datafiltday.loc[j,'daily return of Stock '+str(stock_name)] == 1:
                
            dataallfactorsmod = dataallfactorsmod.drop(j)
            datafiltdaymod = datafiltdaymod.drop(j)

    X = dataallfactorsmod[['Mkt-RF', 'SMB', 'HML','RMW','CMA','Sentiment factor of Stock '+str(stock_name)]]
    y = (datafiltdaymod['daily return of Stock '+str(stock_name)]-1) - dataallfactorsmod['RF']
    X = sm.add_constant(X)
    ff_model = sm.OLS(y, X).fit()
    print(ff_model.summary())
    intercept,b1,b2,b3,b4,b5,b6 = ff_model.params

    for i in range(7):
        regression_values.loc[i,'regression values for Stock '+str(stock_name)] = ff_model.params[i]


regression_values = regression_values.transpose()

regression_values

regression_values.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Topics_Stocks_Regression//All_Topics_Stocks_Regression.csv')

### section to get the regression values without sentiment

# create dataframe for regression values of each stock without sentiment factor
regression_values_minussentiment = pd.DataFrame()
regression_values_minussentiment['regression parameters'] = 'b0','b1','b2','b3','b4','b5'



# regression loop for each stock without the sentiment factor
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    # create copies of dataframes to be modified when certain stock returns aren't available
    datafiltdaymod = datafiltday.copy()
    dataallfactorsmod = dataallfactors.copy()

    # loop to remove rows where sentiment factor doesn't exist yet
    for j in datafiltday.index:

        if datafiltday.loc[j,'daily return of Stock '+str(stock_name)] == 1:
                
            dataallfactorsmod = dataallfactorsmod.drop(j)
            datafiltdaymod = datafiltdaymod.drop(j)

    X = dataallfactorsmod[['Mkt-RF', 'SMB', 'HML','RMW','CMA']]
    y = (datafiltdaymod['daily return of Stock '+str(stock_name)]-1) - dataallfactorsmod['RF']
    X = sm.add_constant(X)
    ff_model = sm.OLS(y, X).fit()
    print(ff_model.summary())
    intercept,b1,b2,b3,b4,b5 = ff_model.params

    for i in range(6):
        regression_values_minussentiment.loc[i,'regression values for Stock '+str(stock_name)] = ff_model.params[i]


regression_values_minussentiment = regression_values_minussentiment.transpose()

regression_values
regression_values_minussentiment

regression_values_minussentiment.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_Topics_Stocks_Regression//All_Topics_Stocks_Regression_minussentiment.csv')

## section to perform an AR(1) sentiment forecast model

# add the sentiment score to the returns 

datafiltday
dataallfactors

# create new dataframe containing the AR(1) data
dataAR1 = pd.DataFrame()

# add a date column
dataAR1['daily date'] = datafiltday['daily date of Stock']

# loop to incorporate the sentiment with the stock return
for i in stock_list:
    print(i)
    stock_name = str(i)

    dataAR1['return + sentiment of Stock '+str(stock_name)] = (datafiltday['daily return of Stock '+str(stock_name)] - 1)+dataallfactors['Sentiment factor of Stock '+str(stock_name)]

dataAR1

# import the AR model
#from statsmodels.tsa.ar_model import AutoReg

#train_data = dataAR1['return + sentiment of Stock GME'][:len(dataAR1)-845]
#test_data = dataAR1['return + sentiment of Stock GME'][len(dataAR1)-845:]

#train_data
#test_data

# fit the AR model with the training data
#ar_model = AutoReg(train_data, lags=1).fit()

## source for these steps : https://vitalflux.com/autoregressive-ar-models-with-python-examples/

#print(ar_model.summary())


#
# Perform the predictions
#
#pred = ar_model.predict(start=len(train_data), end=(len(dataAR1)-400), dynamic=False)

#pred

#pred.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testingg.csv')

#
# Plot the prediction vs test data
#
#plt.plot(pred)
#plt.plot(test_data, color='red')

#plt.show()

## Section with SVR for classification

# https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d 

# Create a dataframe with returns and sentiment factor



# add the corresponding sentiment score with the returns

DataSVR = pd.DataFrame()

# add the corresponding sentiment score with the returns
# loop to only include values that containing a sentiment score as a stacked dataset
for i in stock_list:
    print(i)
    stock_name = str(i)

    for k in range(5):

        for j in range(1,dataallfactors.shape[0]):
            if datafiltday.loc[j,'daily return of Stock '+str(stock_name)] != 1:
                if dataallfactors.loc[j,'Sentiment factor of Stock '+str(stock_name)] != 0:
                    DataSVR.loc[dataallfactors.shape[0]*k+j,'daily return'] = datafiltday.loc[j,'daily return of Stock '+str(stock_name)]
                    DataSVR.loc[dataallfactors.shape[0]*k+j,'sentiment score'] = dataallfactors.loc[j,'Sentiment factor of Stock '+str(stock_name)]
                    DataSVR.loc[dataallfactors.shape[0]*k+j,'stock name'] = str(stock_name)

DataSVR

### do we have to lag the results or not?

# stack the data
# create the X and the Y parts of the dataset

X = DataSVR.iloc[:,1:2].values.astype(float)
y = DataSVR.iloc[:,0:1].values.astype(float)-1


# scale the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

 
DataSVR.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//testinggg.csv')

# create the support vector regressor 
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X,y.ravel())

# predict values

score = regressor.score(X,y)

score

y_pred = regressor.predict(6.5)

y_pred

# Visualising the Support Vector Regression results

plt.scatter(X, y, color = 'magenta')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('SVR Returns based on sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Return')
plt.show()

## hit ratio

predict = []

for i in DataSVR.index: 
    regressor.predict([i])



## looking at feature importance

# https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e

# modify data for random forest

DataRF = pd.DataFrame()

for i in DataSVR.index:
    DataRF.loc[i,'daily return'] = DataSVR.loc[i,'daily return']
    if DataSVR.loc[i,'stock name'] == 'GME':
        DataRF.loc[i,'GME'] = DataSVR.loc[i,'sentiment score']
    if DataSVR.loc[i,'stock name'] == 'AMC':
        DataRF.loc[i,'AMC'] = DataSVR.loc[i,'sentiment score']
    if DataSVR.loc[i,'stock name'] == 'BB':
        DataRF.loc[i,'BB'] = DataSVR.loc[i,'sentiment score']
    if DataSVR.loc[i,'stock name'] == 'NOK':
        DataRF.loc[i,'NOK'] = DataSVR.loc[i,'sentiment score']
    if DataSVR.loc[i,'stock name'] == 'BBBY':
        DataRF.loc[i,'BBBY'] = DataSVR.loc[i,'sentiment score']

# fill na values in new dataframe
DataRF = DataRF.fillna(0)


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
y = DataRF['daily return']
X = pd.DataFrame(DataRF[['GME','AMC','BB','NOK','BBBY']])

np.random.seed(seed = 42)

X['random'] = np.random.random(size = len(X))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
rf.fit(X_train, y_train)

print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train), 
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_valid, y_valid)))

# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,
                       'feature_importance': importances}) \
           .sort_values('feature_importance', ascending = False) \
           .reset_index(drop = True)
    return df

# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    imp_df.columns = ['feature', 'feature_importance']
    sns.barplot(x = 'feature_importance', y = 'feature', data = imp_df, orient = 'h', color = 'royalblue') \
       .set_title(title, fontsize = 20)


base_imp = imp_df(X_train.columns, rf.feature_importances_)
base_imp

var_imp_plot(base_imp, 'Feature importance using random forests')
plt.show()


