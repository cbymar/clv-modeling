import datetime as dt
import matplotlib.pyplot as plt
import lifetimes
import numpy as np
import os
import pandas as pd
import seaborn as sns

def numcard(x):
    return x.nunique(), len(x)
def todateclean(x):
    return pd.to_datetime(x, errors='coerce').dt.date.astype('datetime64')

"""
- info, shape, dtypes
- df.isnull().sum()  #Check for null counts/ value_counts()
- Check for supposed imputed values (are there suspicious values of 0, like for Age. )
- change zeros to nans where appropriate
- Imputation of missing values
- handle stringified json
- df.dtypes # in case obj to (df.colname = df.colname.astype("category"))
- df['colname'] = pd.to_datetime(df['colname']).dt.date
- df.drop("colname", axis=1) # drop columns
- How balanced are the outcomes?  
X = df.drop("diagnosis", axis=1) # just saying which axis again
Y = df["diagnosis"] # this is just a series now

col = X.columns # if we do type(col), it's an Index
X.isnull().sum() # this covers every column in the df.

def rangenorm(x):
    return (x - x.mean())/(x.max() - x.min())
le = LabelEncoder()
le.fit(Y_norm)
"""

df = pd.read_csv("./ignoreland/onlineretail.csv")
df.info()
df.apply(lambda x: numcard(x))

datecols = ['InvoiceDate']
df.loc[:, datecols] = df.loc[:,datecols].apply(lambda x: todateclean(x))

dfnew = df[(df.Quantity>0) & (df.CustomerID.isnull()==False)]
dfnew['amt'] = dfnew['Quantity'] * dfnew['UnitPrice']
dfnew.describe()

from lifetimes.plotting import *
from lifetimes.utils import *
observation_period_end = '2011-12-09'
monetary_value_col = 'amt'
modeldata = summary_data_from_transaction_data(dfnew,
                                               'CustomerID',
                                               'InvoiceDate',
                                               monetary_value_col=monetary_value_col,
                                               observation_period_end=observation_period_end)

modeldata.head()
modeldata.info()  # 4 floats.
# Eyeball distribution of frequency (calculated)
modeldata['frequency'].plot(kind='hist', bins=50)
print(modeldata['frequency'].describe())
print(modeldata['recency'].describe())
print(sum(modeldata['frequency'] == 0)/float(len(modeldata)))

##### Lec21
from lifetimes import BetaGeoFitter
# similar to lifelines
bgf = BetaGeoFitter(penalizer_coef=0.0)  # no regularization param.

bgf.fit(modeldata['frequency'], modeldata['recency'], modeldata['T'])
print(bgf)
# See https://www.youtube.com/watch?v=guj2gVEEx4s and
# https://www.youtube.com/watch?v=gx6oHqpRgpY
## residual lifetime value is more useful construct

from lifetimes.plotting import plot_frequency_recency_matrix
plot_frequency_recency_matrix(bgf)
from lifetimes.plotting import plot_probability_alive_matrix
plot_probability_alive_matrix(bgf)

# lec 24:
# set an outer time boundary and predict cumulative purchases by that time
t = 10 # from now until now+t periods
modeldata['predicted_purchases'] = \
    bgf.conditional_expected_number_of_purchases_up_to_time(t,
                                                            modeldata['frequency'],
                                                            modeldata['recency'],
                                                            modeldata['T'])
modeldata.sort_values(by='predicted_purchases').tail(5)
modeldata.sort_values(by='predicted_purchases').head(5)
# lec 25: validation of model
from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf) # this plot shows very clearly the model performance
# in terms of transaction volume fit

# Lec 26: splitting into train and test (by time period)
summary_cal_holdout = calibration_and_holdout_data(df,
                                                   'CustomerID',
                                                   'InvoiceDate',
                                                   calibration_period_end='2011-06-08',
                                                   observation_period_end='2011-12-09')

summary_cal_holdout.head()

bgf.fit(summary_cal_holdout['frequency_cal'],
        summary_cal_holdout['recency_cal'],
        summary_cal_holdout['T_cal'])

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

plot_calibration_purchases_vs_holdout_purchases(bgf, summary_cal_holdout)

from lifetimes.plotting import plot_history_alive


days_since_birth = 365
fig = plt.figure(figsize=(12,8))
id = 14621  # choose a customer id
sp_trans = df.loc[df['CustomerID'] == id]  # specific customer's covariates
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')

# Lec28: Subsetting to customers who repurchase.
returning_customers_summary = modeldata[modeldata['frequency']>0]
returning_customers_summary.head()
returning_customers_summary.shape
# Lec 29: gamma-gamma model for LTV
# Note: good practice to confirm small/no apparent corr for frequency and mean trxn value
# Rev per trxn: predict total monetary value.
# The Beta param for the gamma model of total spend is itself assumed gamma distributed
# that is where the name comes from.
# teh expectation of total spend for person i is calculated in empirical-bayes fashion, as a weighted
# mean of population average and the sample mean for person i.
# https://antonsruberts.github.io/lifetimes-CLV/ also great additional code.
# Output of ggf fitter:
# p =
# q =
# v =
from lifetimes import GammaGammaFitter
ggf = GammaGammaFitter(penalizer_coef=0.0)

ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
ggf.summary
ggf.conditional_expected_average_profit(modeldata['frequency'],
                                        modeldata['monetary_value'])
# cond_exp_avg_profit => gives prediction of mean trxn value.








###############
# review, per documentation:
bgf.summary
# r, alpha = shape, scale for gamma dist that represents sum (convolution) of purchase rates
# a = alpha param for beta dist of churn
# b = beta param for beta dist of churn
x  = np.random.gamma(.784, 49.28,10000) # r, alpha, n
bgf.summary.loc["a",:][0]/ (bgf.summary.loc["b",:][0] + bgf.summary.loc["a",:][0])



"""
frequency = number of periods in which a non-first purchase was made
T = age in same units of each customer
recency = period[last purchase] - period[first purchase]
monetary_value = sum(money)/(frequency+1)

# use utility functions to aggregate into useable format.
# https://lifetimes.readthedocs.io/en/latest/More%20examples%20and%20recipes.html
# sql examples for aggregating into RFM and doing holdout split.
"""


"""
Also, per brucehardie,
The integrated (function of 2 functions) nature of these problems yields to 
The gaussian hypergeometric function trick for evaluating the double integral.
"""
