#%% Imports
import pandas as pd
import matplotlib.pyplot as plt
import os

from numpy import log
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from STATICS import CHARTS_LOCATION, FILE, LAGS
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from numpy import inf

#%% Clear results directory to avoid appending to existing files
def cleanup():
    directory = 'results/'
    files_to_remove = os.listdir(directory)
    for file in files_to_remove:
        os.remove(directory + file)
    return

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

#%% Function to check stationarity
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
        time_series = data[ticker].dropna()
        log_rtn = log_returns[ticker]
        log_rtn.dropna(inplace=True)
        # TO DO : implement the stationarity test in the title of the multichart
        if is_white_noise(log_rtn):
            figure_name = ticker.upper() + \
                " is white noise.\nNb lags = " + str(lags)
        else:
            figure_name = ticker.upper() + \
                " is NOT white noise.\nNb lags = " + str(lags)
        adf = ADF_test(time_series)
        if adf[3]:
            figure_name = figure_name + \
                "\nThe time series is stationary. ADF stat = " + \
                str(round(adf[0], 3))
        else:
            figure_name = figure_name + \
                "\nThe time series is NOT stationary. ADF stat = " + \
                str(round(adf[0], 3))
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

#%% Tests and returns only time-series featuring some level of autocorrelation
def is_fit_for_AR(data):
    tickers = get_tickers(data)
    log_returns = to_log_return(data)
    not_white_noise = []
    white_noise = []
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()
        if not is_white_noise(log_rtn, nlags=LAGS, thres=0.05):
            not_white_noise.append(ticker)
        else:
            white_noise.append(ticker)
    print("The following tickers are not white noise, \
    and are fit for AR model:", not_white_noise)
    return not_white_noise, white_noise

#%% Finds best order (p) for each time series, and fits an AR(p) model. Prints a summary
def AR_model(data, tickers_with_AR):
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    summary = {}
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()
        model = AR(log_rtn)
        best_order = model.select_order(maxlag=LAGS, ic='aic')
        result =  model.fit(best_order)
        aic = result.aic
        summary[ticker] = [best_order, aic]
    with open('results/AR_models.txt', 'w') as results_file:
        for k in summary.keys():
            message = "Ticker: {} \t\t\tBest order: {} \tAIC = {}\n".format(k, summary[k][0], summary[k][1])
            results_file.write(message)
    return summary

#%% For tickers that don't feature and autocorrelation, we fit an MA model.
def MA_model(data, tickers_without_AR):
    tickers = tickers_without_AR
    log_returns = to_log_return(data)
    summary = {}
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()
        lowest_aic = inf
        for order in range(1,LAGS):
            model = ARMA(log_rtn, order=(0,order))
            result = model.fit()
            result.summary()
            aic = result.aic
            if aic < lowest_aic:
                lowest_aic = aic
                best_order = order
                summary_best_fit = result.summary()
        with open('results/MA_models.txt', 'a') as results_file:
            message = "\n\n\n\t\t\t\t\t********** {} **********\n \
                Best fit for {} obtained with model MA({})\n{}\n" \
                .format(ticker.upper(), ticker, best_order, summary_best_fit)
            results_file.write(message)
    return

#%% ARIMA modelling
def ARIMA_model(data, tickers_with_AR):
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()

#%% SARIMA modelling
def SARIMA_model(data, tickers_with_AR):
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()

#%% ARIMAX modelling
def ARIMAX_model(data, tickers_with_AR):
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    for ticker in tickers:
        endog = log_returns[ticker].dropna()
        exog = tickers.pop(ticker)
#%% SARIMAX modelling
def SARIMAX_model(data, tickers_with_AR):
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()



#%% Tests for ARCH effect (serial autocorrelation of the volatility) in the log-returns
def is_fit_for_ARCH(data):
    tickers = get_tickers(data)
    log_returns = to_log_return(data)
    not_white_noise = []
    for ticker in tickers:
        log_rtn = log_returns[ticker].dropna()
        at2 = ((log_rtn - log_rtn.mean()) ** 2).dropna()
        if not is_white_noise(at2, nlags=LAGS, thres=0.05):
            not_white_noise.append(ticker)
    with open('results/ARCH_models.txt', 'a') as f:
        message = "The following tickers are relevant for ARCH modeling:\n{}".format(not_white_noise)
        f.write(message)
    return not_white_noise

def volatility_model(data, ticker, vol, p, q):
    log_returns = to_log_return(data)
    # tickers = test_for_ARCH
    log_rtn = log_returns[ticker].dropna()
    if ticker == 'vix':
        log_rtn = log_rtn * 100 # rescaling as advised by optimizer
    else:    
        log_rtn = log_rtn * 1000 # rescaling as advised by optimizer
    am = arch_model(log_rtn, vol=vol, p=p, q=q, dist='Normal')
    res = am.fit()
    # print(res.summary())
    return res

def arch_fitting(data, ticker):
    results_table = []
    lowest_aic = inf
    for vol in ['arch', 'garch', 'egarch']:
        for p in range(1, 3):
            for q in range(3):
                res = volatility_model(data, ticker, vol, p, q)
                aic = res.aic
                summary = res.summary()
                if vol == 'arch':
                    results_table.append([ticker, vol, p, aic])
                    if aic < lowest_aic:   
                        best_fit = [ticker, vol, p, aic]
                        lowest_aic = aic
                        best_fit_summary = summary
                else:
                    results_table.append([ticker, vol, (p, q), aic])
                    if aic < lowest_aic:
                        best_fit = [ticker, vol, (p, q), aic]
                        lowest_aic = aic
                        best_fit_summary = summary

    for result in results_table:
        print(result)
    message = '\n\n\n\t\t\t\t\t********** {} **********\n \
                Vol model minimizing AIC for {} is {} with minimum AIC= {}\n{}' \
                .format(ticker.upper(), best_fit[0], best_fit[1:3], best_fit[3], best_fit_summary)
    with open('results/ARCH_models.txt', 'a') as f:
        f.write(message)
    # input('Press <enter> to continue')
    return
# %%
