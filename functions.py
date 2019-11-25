#%% Imports
import pandas as pd
import matplotlib.pyplot as plt

from numpy import log
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from STATICS import CHARTS_LOCATION, FILE, LAGS
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import adfuller


#%% Functions to prepare the data:
# 1/ load the CSV, 
# 2/ get the list of tickers present in the data, 
# 3/ convert the data to log-returns
# 4/ clean-up resulting NaN
# NB: some data is only available at later dates, so it is
# normal to have NaN in certain columns
def load_data(file : str) -> pd.DataFrame:
    # Import csv in DataFrame
    data = pd.read_csv(file)
    new_labels = []
    for label in data.columns:
        label = label.split()[0].lower()
        new_labels.append(label)
    data.columns = new_labels
    data.dates = pd.to_datetime(data.dates, format='%d/%m/%Y')

    data.set_index('dates', inplace=True)
    return data

def get_tickers(df):
    tickers = df.columns
    return tickers

def to_log_return(data):    # Converts all data to log-returns in one single dataframe
    tickers = get_tickers(data)
    for ticker in tickers:
        # Translating tickers featuring negative values
        # https://blogs.sas.com/content/iml/2011/04/27/log-transformations-how-to-handle-negative-data-values.html
        if data[ticker].min() <= 0:
            min = data[ticker].min()
            data[ticker] = data[ticker].apply(lambda x : x + 1 - min)
    log_returns = log(data/data.shift(1))
    log_returns.dropna(axis='index', how='all', inplace=True)
    return log_returns

#%% Functions to check stationarity
def ADF_test(ts):
    # https://machinelearningmastery.com/time-series-data-stationary-python/
    # lags?
    # Null hypothesis: Time series has a unit root => not stationary
    # p-value below threshold <=> rejecting Null hypothesis
    result = adfuller(ts)
    adf_statistic = result[0]
    p_value = result[1]
    five_pct_critical_value = result[4]['5%']
    if p_value < 0.05:
        is_stationary = True
    else:
        is_stationary = False
    return (adf_statistic, p_value, five_pct_critical_value, is_stationary)

# def test_stationarity(data):
#     tickers = get_tickers(data)
#     for ticker in tickers:
#         ts = data[ticker].dropna()




#%% Functions to test autocorrelation in the log_returns of the data
def is_white_noise( log_rtn, \
                    nlags=LAGS, \
                    thres=0.05) -> bool:
    p_values = acf(log_rtn, nlags, qstat=True)[-1]
    pv_above_threshold = [pv for pv in p_values if pv > thres]
    if pv_above_threshold == []:
        is_white_noise = False
    else:
        is_white_noise = True
    return is_white_noise

#%% Function to plot the data for each underlying (=ticker)
def generate_multicharts(data, lags=LAGS):
    tickers = get_tickers(data)
    log_returns = to_log_return(data)
    for ticker in tickers:
        filename = CHARTS_LOCATION + ticker + ".png"
        time_series = data[ticker]
        log_rtn = log_returns[ticker]
        log_rtn.dropna(inplace=True)
        # TO DO : implement the stationarity test in the title of the multichart
        if is_white_noise(log_rtn):
            figure_name = ticker.upper() + \
                " is white noise.\nNb lags = " + str(lags)
        else:
            figure_name = ticker.upper() + \
                " is NOT white noise.\nNb lags = " + str(lags)
        print("Plotting", ticker)
        figsize = (10, 10)
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0))
        logr_ax = plt.subplot2grid(layout, (0, 1))
        acf_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
        pacf_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
        
        ts_ax.plot(time_series)
        logr_ax.plot(log_rtn)
        plot_acf(log_rtn, ax=acf_ax, lags=lags, alpha=.05)
        plot_pacf(log_rtn, ax=pacf_ax, lags=lags, alpha=.05)
        
        fig.suptitle(figure_name, fontsize=20)
        plt.savefig(filename)
        # fig.show()
    return

# %%
def fit_for_AR(data):
    tickers = get_tickers(data)
    log_returns = to_log_return(data)
    not_white_noise = []
    for ticker in tickers:
        log_rtn = log_returns[ticker]
        if not is_white_noise(log_rtn, nlags=LAGS, thres=0.05):
            not_white_noise.append(ticker)
    return not_white_noise
