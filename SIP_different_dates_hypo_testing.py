#..Packages and Modules
import pandas as pd
import numpy as np
from calendar import monthrange
from scipy.optimize import newton
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#..Functions used
#..Finding XIRR
def xnpv(rate,cashflows):
    chron_order = sorted(cashflows, key = lambda x: x[0])
    t0 = chron_order[0][0]
    return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])

def xirr(cashflows,guess=0.1):
    return newton(lambda r: xnpv(r,cashflows),guess)

def returns(data):
    data['Amt_Invested'] = -100
    data['Units_Invested'] = - (data['Amt_Invested'] / data['Price'])
    data['Redeem_units'] = data['Units_Invested'].rolling(window = 60).sum()
    data['Redeem_units'] = data['Redeem_units'].shift(periods = 1)
    data['Redeem_amt'] = data['Redeem_units'] * data['Price']
    sip_returns = []
    for i in range(60,len(data) - 1):
        cfs = [(data['Dates'][j], -100) for j in range(i - 60, i)]
        cfs = cfs + [(data['Dates'][i], data['Redeem_amt'][i])]
        sip_returns.append(xirr(cfs))
    return(sip_returns)

#..Data Preparations
nifty = pd.read_csv(r'D:\Sampathkumar.AP\Desktop\Work\Blog stuff\1. SIP Dates\Nifty_prices.csv')
nifty['Date'] = pd.to_datetime(nifty['Date'], format = '%d-%m-%Y')

dates = pd.date_range(start= nifty['Date'][0], end= nifty['Date'][len(nifty) - 1], freq='D')
nifty_prices = pd.DataFrame()
nifty_prices['Dates'] = dates
nifty_prices = pd.merge(nifty_prices, nifty, left_on= 'Dates', right_on = 'Date', how = 'left').drop('Date', axis = 1)
nifty_prices.columns = ['Dates', 'Price']
del([nifty, dates])

for i in np.where(nifty_prices['Price'].isna())[0]:
    nifty_prices['Price'].iloc[i] = nifty_prices['Price'].iloc[i - 1]

nifty_prices['Last_Day'] = [monthrange(i.year, i.month)[1] for i in nifty_prices['Dates']]
nifty_prices['Day'] = [i.day for i in nifty_prices['Dates']]
sip_last_day = nifty_prices.loc[nifty_prices['Last_Day'] == nifty_prices['Day'], :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_25th_day = nifty_prices.loc[nifty_prices['Day'] == 25, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_1st_day = nifty_prices.loc[nifty_prices['Day'] == 1, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_5th_day = nifty_prices.loc[nifty_prices['Day'] == 5, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_10th_day = nifty_prices.loc[nifty_prices['Day'] == 10, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_15th_day = nifty_prices.loc[nifty_prices['Day'] == 15, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
sip_20th_day = nifty_prices.loc[nifty_prices['Day'] == 20, :].reset_index(drop = True).drop(['Last_Day', 'Day'], axis = 1)
del([nifty_prices, i])

sip_last_day_returns = returns(sip_last_day)
sip_25th_day_returns = returns(sip_25th_day)
sip_1st_day_returns = returns(sip_1st_day)
sip_5th_day_returns = returns(sip_5th_day)
sip_10th_day_returns = returns(sip_10th_day)
sip_15th_day_returns = returns(sip_15th_day)
sip_20th_day_returns = returns(sip_20th_day)

sip_returns_df = pd.DataFrame({'Last_day':sip_last_day_returns, 'Day_1': sip_1st_day_returns, 'Day_5': sip_5th_day_returns, 'Day_10': sip_10th_day_returns, 'Day_15': sip_15th_day_returns, 'Day_20': sip_20th_day_returns, 'Day_25': sip_25th_day_returns})

#..Viz
sip_returns_df.boxplot(column=['Last_day', 'Day_1', 'Day_5', 'Day_10', 'Day_15', 'Day_20', 'Day_25'], grid=False)
stats_df = pd.DataFrame({'SIP_Returns': sip_returns_df.columns})
stats_df['Mean'] = sip_returns_df.mean().reset_index(drop = True)
stats_df['Variance'] = sip_returns_df.var().reset_index(drop = True)
stats_df['CV'] = sip_returns_df.std().reset_index(drop = True) / stats_df['Mean']
print(stats_df)

#..hypothesis testing
#..1. one sample test of mean- using t tests
hypothesized_mean_returns = 0.09

for i in sip_returns_df.columns:
    print('t-test 1 sample- {} mean returns vs hypothesized returns'.format(i))
    tstat, pvalue = stats.ttest_1samp(sip_returns_df[i], hypothesized_mean_returns)
    if pvalue < 0.05:
        print('rejecting null hypothesis at 95% confidence interval\n')
    else:
        print('not rejecting null hypothesis at 95% confidence interval\n')

#..2. two sample test of mean
for i in range(len(sip_returns_df.columns)):
    print(sip_returns_df.columns[i])
    if i < len(sip_returns_df.columns)-1:
        for j in sip_returns_df.columns[i+1:]:
            print('against {}'.format(j))
            tstat, pvalue = stats.ttest_ind(sip_returns_df.iloc[:,i], sip_returns_df.loc[:,j])
            if pvalue < 0.05:
                print('rejecting null hypothesis at 95% confidence interval\n')
            if pvalue > 0.05:
                print('not rejecting null hypothesis at 95% confidence interval\n')

#..One factor ANOVA
fvalue, pvalue = stats.f_oneway(sip_returns_df['Last_day'], sip_returns_df['Day_1'], sip_returns_df['Day_5'], sip_returns_df['Day_10'], sip_returns_df['Day_15'], sip_returns_df['Day_20'], sip_returns_df['Day_25'])
print(fvalue, pvalue)

sip_returns_df_melt = pd.melt(sip_returns_df, value_vars=['Last_day', 'Day_1', 'Day_5', 'Day_10', 'Day_15', 'Day_20', 'Day_25'])
sip_returns_df_melt.columns = ['SIP_Returns', 'Returns']
model = ols('Returns ~ C(SIP_Returns)', data=sip_returns_df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

model_pair_comp = pairwise_tukeyhsd(endog=sip_returns_df_melt['Returns'], groups=sip_returns_df_melt['SIP_Returns'], alpha=0.05)
print(model_pair_comp)

w, pvalue = stats.bartlett(sip_returns_df['Last_day'], sip_returns_df['Day_1'], sip_returns_df['Day_5'], sip_returns_df['Day_10'], sip_returns_df['Day_15'], sip_returns_df['Day_20'], sip_returns_df['Day_25'])
print(w, pvalue)
