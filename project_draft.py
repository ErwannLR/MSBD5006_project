#%% Imports
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#%% Load data
file = 'closing_prices.csv'
data = pd.read_csv(file)
new_labels = []
for label in data.columns:
    label = label.split()[0].lower()
    new_labels.append(label)
data.columns = new_labels
#data.set_index('dates', inplace=True)
print('data', data.head())
#%% Prepare dataframe for a single asset
col = 'xpd'
df
df = pd.DataFrame(data=data[['dates', col]])
pd.to_datetime(df.dates)
df.dropna(axis='index', how='any', inplace=True)
df['simple_rtn'] = df[col].pct_change()
#df['log_rtn'] = np.log(df[col]) - np.log(df[col].shift(1))
df['log_rtn'] = np.log(df[col] / df[col].shift(1))
df.dropna(axis='index', how='any', inplace=True)
df = df[['dates', 'log_rtn']]
df.dates = pd.to_datetime(df.dates)
print('Showing: '+col+'\n', df.head())
#%% Select one year of data
year = 2002
df[df['dates'].dt.year == year]

#%% Computing ACF & PACF
#acf = acf(df.log_rtn.to_list(), nlags=25, qstat=True, alpha=0.05)
run_acf = acf(df.log_rtn, qstat=True, alpha=0.05)

run_pcaf = pacf(df.log_rtn)

#%% Graph ACF & PACF
plot_acf(df.log_rtn, lags=12)
plot_pacf(df.log_rtn,lags=12)
