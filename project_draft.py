#%% Imports
import pandas as pd
import numpy as np
import datetime as dt
from random import choice
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%% Set statics for testing purposes
file = 'closing_prices.csv'
ticker = 'xpd'

#%% Functions
# Load CSV into Dataframe and do some clean-up
def load_data(file : str) -> pd.DataFrame:
    # Import csv in DataFrame
    data = pd.read_csv(file)
    new_labels = []
    for label in data.columns:
        label = label.split()[0].lower()
        new_labels.append(label)
    data.columns = new_labels
    data.dates = pd.to_datetime(data.dates)
    #data.set_index('dates', inplace=True)
    #print('data', data.head())
    return data

# Prepare Df by converting single ticker to log-returns   
def get_df(file     : str, \
           ticker   : str) -> pd.DataFrame:
    data = load_data(file)
    df = data[['dates', ticker]]
    df['log_rtn'] = np.log(df[ticker] / df[ticker].shift(1))
    df = df[['dates', 'log_rtn']]
    df.dropna(axis='index', how='any', inplace=True)
    return df

# test_H0 "All ACFs coeff are zeros"
def test_H0(df    : pd.DataFrame, \
            nlags : int=40, \
            thres : int=0.05) -> bool:
    p_values = acf(df.log_rtn, nlags, qstat=True)[-1]
    pv_above_threshold = [pv for pv in p_values if pv > thres]
    if pv_above_threshold == []:
        reject_H0 = True
    else:
        reject_H0 = False
    return reject_H0

#%% Select a random period in the data
'''As it stands, there is a risk that there is no 
autocorrelation whatsoever in the tickers, I don't know if this is because
the periods covered are really long or something else, but we might need to try for shorter
timeframes.'''
def snip_random_timeframe(df, pct_data=0.25):
    duration = int(len(df.dates) * pct_data)
    random_start = choice(df.dates)
    end_date = random_start + dt.timedelta(days=duration)
    snipped_df = df[df.dates >= random_start][df.dates < end_date]
    return snipped_df #WRONG: Work in progress, snipped_df dates messed up.

'''
    df[df.dates >= choice(df.dates)]    
    choice(df.dates
        year = 2002
        df[df.dates >= choice(df.dates)]
        df_test = df[df['dates'].dt.year == year]


#%% Computing ACF & PACF
#acf = acf(df.log_rtn.to_list(), nlags=25, qstat=True, alpha=0.05)
test_acf = acf(df.log_rtn, qstat=True, alpha=0.05)

run_pcaf = pacf(df.log_rtn)

#%% Graph ACF & PACF
plot_acf(df.log_rtn, lags=12)
plot_pacf(df.log_rtn,lags=12)
'''