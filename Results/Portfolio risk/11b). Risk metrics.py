# imports needed
###### slight issues with the constrained portfolio, works if you re run the code right after the riskparity class definition
from calendar import week
from cmath import nan, sqrt
from dataclasses import dataclass
from re import T
from tkinter import W
from webbrowser import UnixBrowser
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import datetime as dt
import matplotlib.pyplot as mp
import statsmodels.api as sm
from statsmodels.stats.moment_helpers import cov2corr, corr2cov
import seaborn as sns
import pypfopt
from pypfopt import EfficientFrontier
from scipy.optimize import curve_fit

import scipy.optimize as optimize
from scipy.optimize import LinearConstraint
import warnings
warnings.filterwarnings("ignore")

## import relevant data



covmat = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Covariance_Matrix//Covariance_Matrix.csv')
covmat_minussentiment = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//Covariance_Matrix//Covariance_Matrix_minussentiment.csv')

datafiltweek = pd.read_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//All_stock_Yahoo_Historical_filtered_prices//All_stock_Yahoo_Historical__filtered_prices.csv')

## find the expected returns of each stock

datafiltweek

# list of stocks
stock_list = ['AAPL','MSFT' ,'AMZN','TSLA','GOOG','GOOGL','FB','NVDA','AVGO','PEP','COST','CMCSA','ADBE','CSCO','INTC','TMUS','AMD','TXN','QCOM','AMGN','HON','INTU','AMAT','PYPL','ADP','BKNG','SBUX','MDLZ','ADI','NFLX','CHTR','MU','ISRG','GILD','LRCX','REGN','CSX','VRTX','FISV','ATVI','MRNA','MAR','KLAC','AEP','MRVL','NXPI','PANW','KDP','SNPS','EXC','ASML','FTNT','MNST','KHC','ADSK','ABNB','PAYX','CDNS','ORLY','CTAS','XEL','MCHP','MELI','CTSH','EA','AZN','WBA','ILMN','LULU','DLTR','BIDU','JD','LCID','CRWD','IDXX','FAST','WDAY','PCAR','ROST','ODFL','BIIB','DXCM','EBAY','VRSK','CPRT','ZM','SIRI','DDOG','TEAM','SGEN','ANSS','MTCH','PDD','ALGN','NTES','ZS','VRSN','SWKS','DOCU','SPLK','OKTA','CEG']

# fill Nan with 1 in order to calculate cumulative return of each stock
datafiltweek = datafiltweek.fillna(1)

# loop to obtain cumulative returns of each stock
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    for j in datafiltweek.index:
        if j == 0:
            datafiltweek.loc[j,'Cumulative Return of Stock ' +str(stock_name)] = datafiltweek.loc[j,'weekly return of Stock '+str(stock_name)]
        if j > 0 :
            datafiltweek.loc[j,'Cumulative Return of Stock ' +str(stock_name)] = datafiltweek.loc[j,'weekly return of Stock '+str(stock_name)]*datafiltweek.loc[j-1,'Cumulative Return of Stock ' +str(stock_name)]


# loop to find find the number of active weeks in NASDAQ for each stock
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    counter = 0

    for j in datafiltweek.index:
        if datafiltweek.loc[j,'weekly return of Stock ' +str(stock_name)] == 1:
            counter = counter
        if datafiltweek.loc[j,'weekly return of Stock ' +str(stock_name)] < 1 or datafiltweek.loc[j,'weekly return of Stock ' +str(stock_name)] > 1:
            counter = counter + 1

    datafiltweek['Number of active weeks for Stock '+str(stock_name)] = counter

# loop to find the weekly expected return of each stock
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    datafiltweek['Weekly expected return of Stock ' +str(stock_name)] = datafiltweek.loc[(len(datafiltweek)-1),'Cumulative Return of Stock ' +str(stock_name)]**(1/datafiltweek['Number of active weeks for Stock '+str(stock_name)])

datafiltweek


# loop to find the annualised expected return of each stock
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    datafiltweek['Annualised expected return of Stock ' +str(stock_name)] = datafiltweek['Weekly expected return of Stock ' +str(stock_name)]**(52)

datafiltweek

# create dataframe containing the annualised expected return
dataexpret = pd.DataFrame()

# loop to transpose annualised expected return of all Stocks in the NASDAQ
for i in stock_list:
    stock_name = str(i)
    print(stock_name)

    dataexpret['Annualised expected return of Stock ' +str(stock_name)] = datafiltweek['Annualised expected return of Stock ' +str(stock_name)]

dataexpret.shape

dataexpret





####### calculate the inverse-variance portfolio
# use the def from advances in financial machine learning
covmat = covmat.drop(['Unnamed: 0'], axis=1)
covmat_minussentiment = covmat_minussentiment.drop(['Unnamed: 0'], axis=1)

ivp = 1./np.diag(covmat)

ivp/= ivp.sum()

counter = 0

ivp

# create new dataframe to containing results
ivpexpret = pd.DataFrame() 

for i in stock_list:
    stock_name = str(i)
    print(stock_name)
    print(ivp[counter])

    ivpexpret['expret times ivpweight of Stock ' +str(stock_name)] = dataexpret['Annualised expected return of Stock ' +str(stock_name)]*ivp[counter]

    counter += counter

ivpexpret_portfolio = ivpexpret.sum(axis=1)

# final expected return of inverse volatility portfolio
ivpexpret_portfolio[0]



######  set the variables to perform risk metrics : info from Ganchi 


def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def calculate_active_risk_contribution(w,V, mu, c):
    # function that calculates asset contribution to total risk

    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    vol_MRC = V * np.matrix(w).T

    #RC = np.multiply(vol_MRC,np.matrix(w).T) / sigma

    MRC =  -np.matrix(mu).T + c*vol_MRC/sigma
    # Risk Contribution
    RC =  np.multiply(MRC,np.matrix(w).T)

    return RC

def minimum_deviation_objective(x, pars):
    #args[cov_m,   ref ]
    cov = pars[0]
    ref = pars[1]
    return calculate_portfolio_var( x - ref, cov )

def risk_budget_objective_Roncalli(x, pars):
    # calculate portfolio risk
    V = pars[0]  # covariance table
    r = pars[2]  # expected returns
    c = pars[3]  # constant c
    J =  - np.sum(r.T.dot(x)) + c * np.sqrt(calculate_portfolio_var(x, V))
    return J


def active_risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0] # covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    mu = pars[2]
    c = pars[3]

    #risk_target = np.asmatrix(x_t)
    asset_RC = calculate_active_risk_contribution(x,V, mu,c )
    totoalrisk = np.sum(asset_RC)
    risktarget = totoalrisk*x_t
    #sqerror = (asset_RC - risktarget )**2
    J = sum(np.square(asset_RC - risktarget))[0, 0]


    #pct_RC = asset_RC/ np.sum(asset_RC)
    #J = sum(np.square(pct_RC-risk_target ))[0,0] # sum of squared error
    return J

def risk_budget_objective(x,pars):
    # calculate portfolio risk

    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J


def inverse_vol_objective(x, pars):
    V = pars[0]
    vol = 1/np.sqrt(np.diag(V))
    vol = vol/np.sum(vol)
    return np.sum(np.square(x - vol))

def compute_sharpe_ratio(w,V,mu):
    mu = np.matrix(mu)
    w = np.matrix(w)
    sharpe =  w * mu.T / np.sqrt(calculate_portfolio_var(w, V))
    return sharpe[0][0]

def compute_minimum_variance(w,V,mu):
    mu = np.matrix(mu)
    w = np.matrix(w)
    minimum = (calculate_portfolio_var(w, V))
    return minimum 

def maximum_sharpe_objective(x,pars):
    V = pars[0]
    mu =  pars[2]
    return - compute_sharpe_ratio(x, V, mu)

def minimum_sharpe_objective(x,pars):
    V = pars[0]
    mu =  pars[2]
    return compute_sharpe_ratio(x, V, mu)

def minimum_variance_objective(x,pars):
    V = pars[0]
    mu =  pars[2]
    return compute_minimum_variance(x, V, mu)

def add_synthetic(B, value):
    new_row = np.zeros((1, B.shape[1]))
    B = np.append(B, new_row, axis=0)
    new_col = np.zeros((B.shape[0], 1))
    B = np.append(B, new_col, axis=1)
    B[-1, -1] = value
    return B

class riskParity:
    def __init__(self, bnds, xin, f ,  method = "slsqp", mdl = {"ftol": 1e-16}):
        self.bnds = bnds
        self.xin = xin
        self.f = f
        self.mdl = mdl
        self.method = method

    def factor_constraint(self, x):

        constraint = np.sum( self.x_t*np.log(x / self.x_t ) ) + self.kapa
        return constraint

    def total_weight_constraint(self, x):
        return np.sum(x)-1.0

    def long_only_constraint(self, x):
        return x

    def LimitRelativeMarginalContributionToRisk(self, x ):
        # calculate portfolio risk

        B =  self.args[2]['factor_exposure']
        S =  self.args[2]['factor_covariance']
        R =  self.args[2]['specific_variance']
        factor = self.args[2]['factor']

        lambda_t = np.zeros([R.shape[0], R.shape[0]])
        R_df = np.diag(R)

        opt = []
        for f in factor.keys():
            B_t = add_synthetic(B, 1)
            S_t = add_synthetic(S, S[f, f])
            R_t = add_synthetic(R_df, 0)
            Q_t = B_t.dot(S_t).dot(B_t.T) + R_t
            lambda_used = add_synthetic(lambda_t, (1 + factor[f]))
            f_exp = B[:, f]
            synthetic_weights = x.dot(f_exp)
            x_t = np.append(x, synthetic_weights)
            num = x_t.T.dot(lambda_used.dot(Q_t)).dot(x_t) / (x_t.T.dot(Q_t).dot(x_t))
            opt.append(factor[f] - num)

        return np.array(opt)

    # def minimum_deviation(self, args):
    #     # args [cov_m , ref ]
    #
    #     self.cons = ({'type': 'eq', 'fun': self.total_weight_constraint},
    #                  {'type': 'ineq', 'fun': self.long_only_constraint})
    #     self.args = args
    #     return self


    def standard_constraints(self, args ):
        # args [cov_m ,x_t ]

        self.cons = ({'type':'eq', 'fun': self.total_weight_constraint},
                     {'type':'ineq', 'fun': self.long_only_constraint})
        self.args = args
        return self

    def standard_risk_parity_Roncalli(self, args):
        # args [cov_m ,x_t, r, c, kapa ]

        self.cons = ({'type': 'ineq', 'fun': self.factor_constraint},
                     {'type': 'ineq', 'fun': self.long_only_constraint})
        self.x_t = args[1]
        self.kapa = args[4]
        self.args = args
        return self

    def constrained_risk_parity(self, args ):
        # args [cov_m ,x_t, factor_constraints]
        self.cons = ({'type': 'eq', 'fun': self.total_weight_constraint},
                     {'type': 'ineq', 'fun': self.long_only_constraint},
                     {'type': 'ineq', 'fun': self.LimitRelativeMarginalContributionToRisk})
        self.args = args
        return self

    def run(self):
        sol = optimize.minimize(self.f,
                                self.xin,
                                args = self.args,
                                method = self.method,
                                bounds = self.bnds,
                                constraints = self.cons,
                                options = self.mdl  )
        return sol

# setting parameters for risk metric calculations
mdl ={  "ftol": 1e-16}
x_t = np.ones(np.shape(covmat)[0])*1/np.shape(covmat)[0]
lb = np.zeros(len(covmat))#*0.005
ub = np.ones(len(covmat))*0.02
bnds = np.array((lb,ub)).T
xin = ub/np.sum(ub)
r = np.ones(len(covmat))*0.07
f = risk_budget_objective_Roncalli

dataexpret = dataexpret[:-649]

dataexpret = dataexpret.to_numpy()

dataexpret = dataexpret-1

dataexpret = dataexpret[0]

r = dataexpret

# transform covariance matrix into an array
covmat = covmat.to_numpy()
covmat_minussentiment = covmat_minussentiment.to_numpy()

#### calculate the variance of the inverse volatility portfolio
print((calculate_portfolio_var(ivp,covmat))**0.5)

# calculate the expected return of the inverse volatiltiy portfolio
np.sum(dataexpret*ivp)

## calculate the inverse weighted portfolio
args = [covmat, x_t]
ivp1 = riskParity(bnds, xin, inverse_vol_objective ).standard_constraints(args).run()
print((ivp1.x))
# calculate the portfolio volatility of the risk parity portfolio
print((calculate_portfolio_var(ivp1.x,covmat))**0.5)

# calculate the portofolio expected return of the risk parity portfolio
np.sum(dataexpret*ivp1.x)

## calculate the inverse weighted portfolio without sentiment
args = [covmat_minussentiment, x_t]
ivp1_minussentiment = riskParity(bnds, xin, inverse_vol_objective ).standard_constraints(args).run()
print((ivp1_minussentiment.x))
# calculate the portfolio volatility of the risk parity portfolio
print((calculate_portfolio_var(ivp1_minussentiment.x,covmat_minussentiment))**0.5)

# calculate the portofolio expected return of the risk parity portfolio
np.sum(dataexpret*ivp1_minussentiment.x)


#### calculate the portfolio weights of the risk parity portfolio
args = [covmat, x_t]
sol = riskParity(bnds, xin, risk_budget_objective ).standard_constraints(args).run()
print((sol.x))
print(calculate_risk_contribution(sol.x, covmat))
print(sum(calculate_risk_contribution(sol.x, covmat)))

# calculate the portfolio volatility of the risk parity portfolio
print((calculate_portfolio_var(sol.x,covmat))**0.5)

# calculate the portofolio expected return of the risk parity portfolio
np.sum(dataexpret*sol.x)


#### calculate the portfolio weights of the risk parity portfolio without sentiment
args = [covmat_minussentiment, x_t]
sol_minussentiment = riskParity(bnds, xin, risk_budget_objective ).standard_constraints(args).run()
print((sol_minussentiment.x))
print(calculate_risk_contribution(sol_minussentiment.x, covmat_minussentiment))
print(sum(calculate_risk_contribution(sol_minussentiment.x, covmat_minussentiment)))

# calculate the portfolio volatility of the risk parity portfolio without sentiment
print((calculate_portfolio_var(sol_minussentiment.x,covmat_minussentiment))**0.5)

# calculate the portofolio expected return of the risk parity portfolio without sentiment
np.sum(dataexpret*sol_minussentiment.x)


#### calculate the portfolio weights of the constrained risk parity portfolio
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 1]
solc = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc.x))
print(calculate_active_risk_contribution(solc.x, covmat, r, 1))

# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
np.sum(dataexpret*solc.x)


#### calculate the portfolio weights of the constrained risk parity portfolio without sentiment
bnds = np.array((lb, ub)).T  # [0]
args = [covmat_minussentiment, x_t, r, 1]
solc_minussentiment = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc_minussentiment.x))
print(calculate_active_risk_contribution(solc_minussentiment.x, covmat_minussentiment, r, 1))

# calculate the portfolio volatility of the constrained risk parity portfolio without sentiment
print((calculate_portfolio_var(solc_minussentiment.x,covmat_minussentiment))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio without sentiment
np.sum(dataexpret*solc_minussentiment.x)


# minimum variance of portfolio
args = [covmat, x_t, r, 1]
sol_minvar = riskParity(bnds, xin, minimum_variance_objective).standard_constraints(args).run()
print((sol_minvar.x))
print(compute_minimum_variance(sol_minvar.x,covmat,r))

# calculate the portfolio volatility of the minimum variance portfolio
print((calculate_portfolio_var(sol_minvar.x,covmat))**0.5)

# calculate the portfolio expected return of the minimum variance portfolio
np.sum(dataexpret*sol_minvar.x)


# minimum variance of portfolio without sentiment
args = [covmat_minussentiment, x_t, r, 1]
sol_minvar_minussentiment = riskParity(bnds, xin, minimum_variance_objective).standard_constraints(args).run()
print((sol_minvar_minussentiment.x))
print(compute_minimum_variance(sol_minvar_minussentiment.x,covmat_minussentiment,r))

# calculate the portfolio volatility of the minimum variance portfolio
print((calculate_portfolio_var(sol_minvar_minussentiment.x,covmat_minussentiment))**0.5)

# calculate the portfolio expected return of the minimum variance portfolio
np.sum(dataexpret*sol_minvar_minussentiment.x)


# get maximum sharpe of portfolio
args = [covmat, x_t, r, 1]
sol_max = riskParity(bnds, xin, maximum_sharpe_objective).standard_constraints(args).run()
print((sol_max.x))
print(compute_sharpe_ratio(sol_max.x,covmat,r))

# calculate the portfolio volatility of the maximum sharpe ratio portfolio
print((calculate_portfolio_var(sol_max.x,covmat))**0.5)

# calculate the portofolio expected return of the maximum sharpe ratio portfolio
np.sum(dataexpret*sol_max.x)

# get maximum sharpe of portfolio without sentiment
args = [covmat_minussentiment, x_t, r, 1]
sol_max_minussentiment = riskParity(bnds, xin, maximum_sharpe_objective).standard_constraints(args).run()
print((sol_max_minussentiment.x))
print(compute_sharpe_ratio(sol_max_minussentiment.x,covmat_minussentiment,r))

# calculate the portfolio volatility of the maximum sharpe ratio portfolio
print((calculate_portfolio_var(sol_max_minussentiment.x,covmat_minussentiment))**0.5)

# calculate the portofolio expected return of the maximum sharpe ratio portfolio
np.sum(dataexpret*sol_max_minussentiment.x)




# get minimum sharpe of portfolio
sol_min = riskParity(bnds, xin, minimum_sharpe_objective).standard_constraints(args).run()
print((sol_min.x))
print(compute_sharpe_ratio(sol_min.x, covmat, r))

# calculate the portfolio volatility of the minimum sharpe ratio portfolio
print(calculate_portfolio_var(sol_min.x,covmat))

# calculate the portofolio expected return of the minimum sharpe ratio portfolio
np.sum(dataexpret*sol_min.x)


##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 1]
solc = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc.x))
print(calculate_active_risk_contribution(solc.x, covmat, r, 1))



# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
np.sum(dataexpret*solc.x)

##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 0.9]
solc09 = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc09.x))
print(calculate_active_risk_contribution(solc09.x, covmat, r, 0.9))

# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc09.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
sum(sum(dataexpret*solc09.x))


##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 0.8]
solc08 = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc08.x))
print(calculate_active_risk_contribution(solc08.x, covmat, r, 0.8))

# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc08.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
sum(sum(dataexpret*solc08.x))


##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 0.7]
solc07 = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc07.x))
print(calculate_active_risk_contribution(solc07.x, covmat, r, 0.7))

# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc07.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
sum(sum(dataexpret*solc07.x))

##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 0.6]
solc06 = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc06.x))
print(calculate_active_risk_contribution(solc06.x, covmat, r, 0.6))

# calculate the portfolio volatility of the constrained risk parity portfolio
print((calculate_portfolio_var(solc06.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
sum(sum(dataexpret*solc06.x))

##### repeat previous but with a different lambda
bnds = np.array((lb, ub)).T  # [0]
args = [covmat, x_t, r, 0.5]
solc05 = riskParity(bnds, xin, active_risk_budget_objective).standard_constraints(args).run()
print((solc05.x))
print(calculate_active_risk_contribution(solc05.x, covmat, r, 0.5))

# calculate the portfolio volatility of the constrained risk parity portfolio

print((calculate_portfolio_var(solc05.x,covmat))**0.5)

# calculate the portofolio expected return of the constrained risk parity portfolio
sum(sum(dataexpret*solc05.x))

### section to find weights on the efficient frontier

# https://www.kaggle.com/code/trangthvu/efficient-frontier-optimization/notebook

# https://pyportfolioopt.readthedocs.io/en/latest/MeanVariance.html#module-pypfopt.efficient_frontier 

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from pypfopt import EfficientFrontier

r = np.transpose(r)
covmat.shape


# confirmation that the minimum variance portfolio works
ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))  # setup
z = ef.min_volatility() 
z
w = []
for i in z.keys():
    w+=[[z[i]]]
w = np.array(w)
w = np.transpose(w)
w
print((calculate_portfolio_var(w,covmat))**0.5)
np.sum(dataexpret*w)

# calculate values along efficient frontier
ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02)) 
z = ef.efficient_risk(0.143)

w1 = []
for i in z.keys():
    w1+=[[z[i]]]
w1 = np.array(w1)
w1 = np.transpose(w1)
w1
print((calculate_portfolio_var(w1,covmat))**0.5)
np.sum(dataexpret*w1)


ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.145)

w2 = []
for i in z.keys():
    w2+=[[z[i]]]
w2 = np.array(w2)
w2 = np.transpose(w2)
w2
print((calculate_portfolio_var(w2,covmat))**0.5)
np.sum(dataexpret*w2)

ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.150)

w3 = []
for i in z.keys():
    w3+=[[z[i]]]
w3 = np.array(w3)
w3 = np.transpose(w3)
w3
print((calculate_portfolio_var(w3,covmat))**0.5)
np.sum(dataexpret*w3)

ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.155)

w4 = []
for i in z.keys():
    w4+=[[z[i]]]
w4 = np.array(w4)
w4 = np.transpose(w4)
w4
print((calculate_portfolio_var(w4,covmat))**0.5)
np.sum(dataexpret*w4)

ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.160)

w5 = []
for i in z.keys():
    w5+=[[z[i]]]
w5 = np.array(w5)
w5 = np.transpose(w5)
w5
print((calculate_portfolio_var(w5,covmat))**0.5)
np.sum(dataexpret*w5)

ef = EfficientFrontier(r, covmat,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.165)

w6 = []
for i in z.keys():
    w6+=[[z[i]]]
w6 = np.array(w6)
w6 = np.transpose(w6)
w6
print((calculate_portfolio_var(w6,covmat))**0.5)
np.sum(dataexpret*w6)

# same steps for the efficient frontier without the sentiment factor
# confirmation that the minimum variance portfolio works
ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))  # setup
z = ef.min_volatility() 
z
ww = []
for i in z.keys():
    ww+=[[z[i]]]
ww = np.array(ww)
ww = np.transpose(ww)
ww
print((calculate_portfolio_var(ww,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww)


# calculate values along efficient frontier
ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02)) 
z = ef.efficient_risk(0.145)

ww1 = []
for i in z.keys():
    ww1+=[[z[i]]]
ww1 = np.array(ww1)
ww1 = np.transpose(ww1)
ww1
print((calculate_portfolio_var(ww1,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww1)


ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.147)

ww2 = []
for i in z.keys():
    ww2+=[[z[i]]]
ww2 = np.array(ww2)
ww2 = np.transpose(ww2)
ww2
print((calculate_portfolio_var(ww2,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww2)

ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.150)

ww3 = []
for i in z.keys():
    ww3+=[[z[i]]]
ww3 = np.array(ww3)
ww3 = np.transpose(ww3)
ww3
print((calculate_portfolio_var(ww3,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww3)

ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.155)

ww4 = []
for i in z.keys():
    ww4+=[[z[i]]]
ww4 = np.array(ww4)
ww4 = np.transpose(ww4)
ww4
print((calculate_portfolio_var(ww4,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww4)

ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.160)

ww5 = []
for i in z.keys():
    ww5+=[[z[i]]]
ww5 = np.array(ww5)
ww5 = np.transpose(ww5)
ww5
print((calculate_portfolio_var(ww5,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww5)

ef = EfficientFrontier(r, covmat_minussentiment,weight_bounds=(0,0.02))
z = ef.efficient_risk(0.165)

ww6 = []
for i in z.keys():
    ww6+=[[z[i]]]
ww6 = np.array(ww6)
ww6 = np.transpose(ww6)
ww6
print((calculate_portfolio_var(ww6,covmat_minussentiment))**0.5)
np.sum(dataexpret*ww6)


#bounds = Bounds(0, 1)

#dataexpret.shape

#w_min = sol_minvar.x
#w = w_min

#w_sharpe = sol_max.x
#num_ports = 100
#gap = (np.amax(dataexpret) - dataexpret.dot(w))/num_ports

#all_weights = np.zeros((num_ports, 102))
#all_weights[0],all_weights[1]=w,w_sharpe
#ret_arr = np.zeros(num_ports)
#ret_arr[0],ret_arr[1]=dataexpret.dot(w), dataexpret.dot(w_sharpe)
#vol_arr = np.zeros(num_ports)
#vol_arr[0],vol_arr[1]=((calculate_portfolio_var(w,covmat))**0.5),((calculate_portfolio_var(w_sharpe,covmat))**0.5)


# for i in range(num_ports):
    #port_ret = dataexpret.dot(w) + i*gap
    #double_constraint = LinearConstraint([np.ones((dataexpret.shape[1],102))],[1,port_ret],[1,port_ret])


    #Create x0: initial guesses for weights.
    # x0 = np.reshape(sol_minvar.x,(102,1))
    #x0 = w_min
    #Define a function for portfolio volatility.
    #fun = lambda w: np.sqrt(np.dot(w,np.dot(w,covmat)))
    #a = minimize(fun,x0,method='trust-constr',constraints = double_constraint,bounds = bounds)
    


    #all_weights[i,:]=a.x
    #ret_arr[i]=port_ret
    #vol_arr[i]=((calculate_portfolio_var(a.x,covmat))**0.5)

#sharpe_arr = ret_arr/vol_arr  

#mp.figure(figsize=(20,10))
#mp.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
#mp.show()

### simulate multiple portfolios to obtain efficient frontier

# create dataframe containing portfolio results including the sentiment factor 
portfolio_mean_var = pd.DataFrame()

portfolio_mean_var.loc[0,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_max.x,covmat))**0.5)
portfolio_mean_var.loc[0,'Portfolio Mean'] = np.sum(dataexpret*sol_max.x)

portfolio_mean_var.loc[1,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minvar.x,covmat))**0.5)
portfolio_mean_var.loc[1,'Portfolio Mean'] = np.sum(dataexpret*sol_minvar.x)

portfolio_mean_var.loc[2,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol.x,covmat))**0.5)
portfolio_mean_var.loc[2,'Portfolio Mean'] = np.sum(dataexpret*sol.x)

# portfolio_mean_var.loc[3,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(solc.x,covmat))**0.5)
# portfolio_mean_var.loc[3,'Portfolio Mean'] = sum(sum(dataexpret*solc.x))

portfolio_mean_var.loc[3,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ivp1.x,covmat))**0.5)
portfolio_mean_var.loc[3,'Portfolio Mean'] = np.sum(dataexpret*ivp1.x)

portfolio_mean_var.loc[4,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w1,covmat))**0.5)
portfolio_mean_var.loc[4,'Portfolio Mean'] = np.sum(dataexpret*w1)

portfolio_mean_var.loc[5,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w2,covmat))**0.5)
portfolio_mean_var.loc[5,'Portfolio Mean'] = np.sum(dataexpret*w2)

portfolio_mean_var.loc[6,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w3,covmat))**0.5)
portfolio_mean_var.loc[6,'Portfolio Mean'] = np.sum(dataexpret*w3)

portfolio_mean_var.loc[7,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w4,covmat))**0.5)
portfolio_mean_var.loc[7,'Portfolio Mean'] = np.sum(dataexpret*w4)

portfolio_mean_var.loc[8,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w5,covmat))**0.5)
portfolio_mean_var.loc[8,'Portfolio Mean'] = np.sum(dataexpret*w5)

portfolio_mean_var.loc[9,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w6,covmat))**0.5)
portfolio_mean_var.loc[9,'Portfolio Mean'] = np.sum(dataexpret*w6)

portfolio_mean_var

# truncate a part of the dataframe to create efficient frontier
portfolio_mean_var_efficient_frontier = portfolio_mean_var.drop(portfolio_mean_var.index[[ 0,2,3 ]])
portfolio_mean_var_efficient_frontier


colors = np.array(['red','red','green','brown','blue','blue','blue','blue','blue','blue'])
portfolio_mean_var.plot(x='Portfolio Standard Deviation',y='Portfolio Mean', c=colors, kind='scatter')
mp.plot(portfolio_mean_var_efficient_frontier['Portfolio Standard Deviation'],portfolio_mean_var_efficient_frontier['Portfolio Mean'])

mp.title('Portfolio results with the sentiment factor included')
mp.xlabel('Standard deviation')
mp.ylabel('Mean')
mp.show()



## section to display the results excluding the sentiment factor
portfolio_mean_var_minussentiment = pd.DataFrame()

portfolio_mean_var_minussentiment.loc[0,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_max_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[0,'Portfolio Mean'] = np.sum(dataexpret*sol_max_minussentiment.x)

portfolio_mean_var_minussentiment.loc[1,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minvar_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[1,'Portfolio Mean'] = np.sum(dataexpret*sol_minvar_minussentiment.x)

portfolio_mean_var_minussentiment.loc[2,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[2,'Portfolio Mean'] = np.sum(dataexpret*sol_minussentiment.x)

# portfolio_mean_var_minussentiment.loc[3,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(solc_minussentiment.x,covmat_minussentiment))**0.5)
# portfolio_mean_var_minussentiment.loc[3,'Portfolio Mean'] = sum(sum(dataexpret*solc_minussentiment.x))

portfolio_mean_var_minussentiment.loc[3,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ivp1_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[3,'Portfolio Mean'] = np.sum(dataexpret*ivp1_minussentiment.x)

portfolio_mean_var_minussentiment.loc[4,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww1,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[4,'Portfolio Mean'] = np.sum(dataexpret*ww1)

portfolio_mean_var_minussentiment.loc[5,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww2,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[5,'Portfolio Mean'] = np.sum(dataexpret*ww2)

portfolio_mean_var_minussentiment.loc[6,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww3,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[6,'Portfolio Mean'] = np.sum(dataexpret*ww3)

portfolio_mean_var_minussentiment.loc[7,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww4,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[7,'Portfolio Mean'] = np.sum(dataexpret*ww4)

portfolio_mean_var_minussentiment.loc[8,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww5,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[8,'Portfolio Mean'] = np.sum(dataexpret*ww5)

portfolio_mean_var_minussentiment.loc[9,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww6,covmat_minussentiment))**0.5)
portfolio_mean_var_minussentiment.loc[9,'Portfolio Mean'] = np.sum(dataexpret*ww6)

portfolio_mean_var_minussentiment

# truncate a part of the dataframe to create efficient frontier
portfolio_mean_var_minussentiment_efficient_frontier = portfolio_mean_var_minussentiment.drop(portfolio_mean_var_minussentiment.index[[ 0,2,3 ]])
portfolio_mean_var_minussentiment_efficient_frontier

colors = np.array(['red','red','green','brown','blue','blue','blue','blue','blue','blue'])
portfolio_mean_var_minussentiment.plot(x='Portfolio Standard Deviation',y='Portfolio Mean',c=colors, kind='scatter')

mp.plot(portfolio_mean_var_minussentiment_efficient_frontier['Portfolio Standard Deviation'],portfolio_mean_var_minussentiment_efficient_frontier['Portfolio Mean'])
mp.title('Portfolio results without the sentiment factor included')
mp.xlabel('Standard deviation')
mp.ylabel('Mean')
mp.show()

## section to compare both plots
portfolio_mean_var_comparison = pd.DataFrame()

portfolio_mean_var_comparison.loc[0,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_max.x,covmat))**0.5)
portfolio_mean_var_comparison.loc[0,'Portfolio Mean'] = np.sum(dataexpret*sol_max.x)

portfolio_mean_var_comparison.loc[1,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minvar.x,covmat))**0.5)
portfolio_mean_var_comparison.loc[1,'Portfolio Mean'] = np.sum(dataexpret*sol_minvar.x)

portfolio_mean_var_comparison.loc[2,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol.x,covmat))**0.5)
portfolio_mean_var_comparison.loc[2,'Portfolio Mean'] = np.sum(dataexpret*sol.x)

#portfolio_mean_var_comparison.loc[3,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(solc.x,covmat))**0.5)
#portfolio_mean_var_comparison.loc[3,'Portfolio Mean'] = sum(sum(dataexpret*solc.x))

portfolio_mean_var_comparison.loc[4,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ivp1.x,covmat))**0.5)
portfolio_mean_var_comparison.loc[4,'Portfolio Mean'] = np.sum(dataexpret*ivp1.x)

portfolio_mean_var_comparison.loc[5,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w1,covmat))**0.5)
portfolio_mean_var_comparison.loc[5,'Portfolio Mean'] = np.sum(dataexpret*w1)

portfolio_mean_var_comparison.loc[6,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w2,covmat))**0.5)
portfolio_mean_var_comparison.loc[6,'Portfolio Mean'] = np.sum(dataexpret*w2)

portfolio_mean_var_comparison.loc[7,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w3,covmat))**0.5)
portfolio_mean_var_comparison.loc[7,'Portfolio Mean'] = np.sum(dataexpret*w3)

portfolio_mean_var_comparison.loc[8,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w4,covmat))**0.5)
portfolio_mean_var_comparison.loc[8,'Portfolio Mean'] = np.sum(dataexpret*w4)

portfolio_mean_var_comparison.loc[9,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w5,covmat))**0.5)
portfolio_mean_var_comparison.loc[9,'Portfolio Mean'] = np.sum(dataexpret*w5)

portfolio_mean_var_comparison.loc[10,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(w6,covmat))**0.5)
portfolio_mean_var_comparison.loc[10,'Portfolio Mean'] = np.sum(dataexpret*w6)

portfolio_mean_var_comparison.loc[11,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_max_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[11,'Portfolio Mean'] = np.sum(dataexpret*sol_max_minussentiment.x)

portfolio_mean_var_comparison.loc[12,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minvar_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[12,'Portfolio Mean'] = np.sum(dataexpret*sol_minvar_minussentiment.x)

portfolio_mean_var_comparison.loc[13,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(sol_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[13,'Portfolio Mean'] = np.sum(dataexpret*sol_minussentiment.x)

# portfolio_mean_var_comparison.loc[14,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(solc_minussentiment.x,covmat_minussentiment))**0.5)
# portfolio_mean_var_comparison.loc[14,'Portfolio Mean'] = sum(sum(dataexpret*solc_minussentiment.x))

portfolio_mean_var_comparison.loc[15,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ivp1_minussentiment.x,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[15,'Portfolio Mean'] = np.sum(dataexpret*ivp1_minussentiment.x)

portfolio_mean_var_comparison.loc[16,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww1,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[16,'Portfolio Mean'] = np.sum(dataexpret*ww1)

portfolio_mean_var_comparison.loc[17,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww2,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[17,'Portfolio Mean'] = np.sum(dataexpret*ww2)

portfolio_mean_var_comparison.loc[18,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww3,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[18,'Portfolio Mean'] = np.sum(dataexpret*ww3)

portfolio_mean_var_comparison.loc[19,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww4,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[19,'Portfolio Mean'] = np.sum(dataexpret*ww4)

portfolio_mean_var_comparison.loc[20,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww5,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[20,'Portfolio Mean'] = np.sum(dataexpret*ww5)

portfolio_mean_var_comparison.loc[21,'Portfolio Standard Deviation'] = ((calculate_portfolio_var(ww6,covmat_minussentiment))**0.5)
portfolio_mean_var_comparison.loc[21,'Portfolio Mean'] = np.sum(dataexpret*ww6)

portfolio_mean_var_comparison

colors = np.array(['blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','red','red','red','red','red','red','red','red','red','red'])

portfolio_mean_var_comparison.plot(x='Portfolio Standard Deviation',y='Portfolio Mean',c=colors, kind='scatter')
mp.plot(portfolio_mean_var_minussentiment_efficient_frontier['Portfolio Standard Deviation'],portfolio_mean_var_minussentiment_efficient_frontier['Portfolio Mean'])
mp.plot(portfolio_mean_var_efficient_frontier['Portfolio Standard Deviation'],portfolio_mean_var_efficient_frontier['Portfolio Mean'])
mp.title('Comparison of portfolio results with and without the sentiment factor')
mp.xlabel('Standard deviation')
mp.ylabel('Mean')
mp.show()

## section to show the correlation matrix
covmat.shape

corrmat = np.corrcoef(covmat,rowvar=False)

corrmat

np.round(covmat,2)

## section to find the difference between both correlations with and without sentiment

corrmat_minussentiment = np.corrcoef(covmat_minussentiment,rowvar=False)

corrmat_minussentiment

diff_corrmat = corrmat-corrmat_minussentiment

diff_corrmat

# hierarchical clustering 
## https://www.kaggle.com/code/sgalella/correlation-heatmaps-with-hierarchical-clustering/notebook

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

mp.figure(figsize=(15,10))

sns.heatmap(np.round(diff_corrmat,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1)

mp.show()

tester = np.round(diff_corrmat,4)

## dendogram section
mp.figure(figsize=(12,5))
dissimilarity = 1 - abs(tester)

Z = linkage(squareform(dissimilarity), 'complete')


dendrogram(Z, labels=stock_list, orientation='top', 
           leaf_rotation=90);

mp.show()


#clusterize the data
threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
labels

# Keep the indices to sort labels
labels_order = np.argsort(labels)

labels_order

labels_order = pd.DataFrame(labels_order)

labels_order

labels_order.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//labels_order//labels_order.csv')

## same steps as above but for the covariance matrix without sentiment

## section to show the covariance matrix
covmat_minussentiment.shape

corrmat_minussentiment = np.corrcoef(covmat_minussentiment,rowvar=False)

corrmat_minussentiment

np.round(covmat_minussentiment,2)

# hierarchical cluestering
#mp.figure(figsize=(15,10))

#sns.heatmap(np.round(corrmat_minussentiment,2), cmap='RdBu', annot=True, 
            #annot_kws={"size": 7}, vmin=-1, vmax=1)

#mp.show()

tester = np.round(corrmat_minussentiment,4)

## dendogram section
mp.figure(figsize=(12,5))
dissimilarity = 1 - abs(tester)

Z = linkage(squareform(dissimilarity), 'complete')


dendrogram(Z, labels=stock_list, orientation='top', 
           leaf_rotation=90);

mp.show()


#clusterize the data
threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
labels

# Keep the indices to sort labels
labels_order_minussentiment = np.argsort(labels)

labels_order_minussentiment

labels_order_minussentiment = pd.DataFrame(labels_order_minussentiment)

labels_order_minussentiment
labels_order

labels_order_minussentiment.to_csv('C://Users//ijnuh//Desktop//JSV//Bayes//Term3//Project_AlternativeData//Python//labels_order//labels_order_minussentiment.csv')

###### build the new dataframe in a file 

# Build a new dataframe with the sorted columns
for idx, i in enumerate(stock_list[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(np.array(stock_list)[i])
    else:
        df_to_append = pd.DataFrame(np.array(stock_list)[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)



# fig = mp.figure(figsize=(36,36), dpi = 480)
hm = sns.heatmap(covmat-covmat_minussentiment,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)

mp.show()

# fig = mp.figure(figsize=(36,36), dpi = 480)
hm = sns.heatmap(corrmat,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)

mp.show()

# fig = mp.figure(figsize=(36,36), dpi = 480)
hm = sns.heatmap(corrmat_minussentiment,#cbar=True,
                 # annot=True,
                 #square=True,
                 #fmt='.2f',
                 #annot_kws={'size': 12},
                 #cmap='coolwarm',                 
)

mp.show()

