# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import os

from numpy import log
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from STATICS import CHARTS_LOCATION, FILE, LAGS
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from numpy import inf
from numpy import sqrt

#pd.plotting.register_matplotlib_converters()

# %% Clear results directory to avoid appending to existing files
def cleanup():
    directory = 'results/'
    files_to_remove = os.listdir(directory)
    for file in files_to_remove:
        os.remove(directory + file)
    return


# %% Functions to prepare the data:
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


def to_log_return(data):  # Converts all data to log-returns in one single dataframe
    tickers = get_tickers(data)
    for ticker in tickers:
        # Translating tickers featuring negative values
        # https://blogs.sas.com/content/iml/2011/04/27/log-transformations-how-to-handle-negative-data-values.html
        if data[ticker].min() <= 0:
            min = data[ticker].min()
            data[ticker] = data[ticker].apply(lambda x : x + 1 - min)
    log_returns = log(data / data.shift(1))
    log_returns.dropna(axis='index', how='all', inplace=True)
    return log_returns


# %% Function to check stationarity
def ADF_test(ts):
    print("ADF_test() start execute")
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


# %% Functions to test autocorrelation in the log_returns of the data
def is_white_noise(log_rtn, \
                   *ticker, \
                   nlags=LAGS, \
                   thres=0.05) -> bool:
    print("is_white_noise() start execute")
    p_values = acf(log_rtn, nlags, qstat=True)[-1]
    # print('Largest p-value for', ticker[0].upper(), max(p_values))
    pv_above_threshold = [pv for pv in p_values if pv > thres]
    if pv_above_threshold == []:
        is_white_noise = False
    else:
        is_white_noise = True
    with open('results/p_values.txt', 'a') as f:
        message = '\nLargest p-value for ' + str(ticker[0].upper()) + ':\t' + str(max(p_values)) + '\tWhite noise: ' + str(is_white_noise)
        f.write(message)
    return is_white_noise


# %% Function to plot the data for each underlying (=ticker)
def generate_multicharts(data, lags=LAGS):
    print("generate_multicharts() start execute")
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


# %% Tests and returns only time-series featuring some level of autocorrelation
def is_fit_for_AR(log_returns):
    print("is_fit_for_AR() start execute")
    tickers = get_tickers(log_returns)
    #log_returns = to_log_return(data)
    not_white_noise = []
    white_noise = []
    for ticker in tickers:
        print("is_fit_for_AR() start execute in ticker: " + ticker)
        log_rtn = log_returns[ticker].dropna()
        if not is_white_noise(log_rtn, ticker, nlags=LAGS, thres=0.05):
            not_white_noise.append(ticker)
        else:
            white_noise.append(ticker)
    print("The following tickers are not white noise, \
    and are fit for AR model:", not_white_noise)
    print("The following tickers are white noise series, \
    and need to check for ARCH effect:", white_noise)
    return not_white_noise, white_noise


# %% Finds best order (p) for each time series, and fits an AR(p) model. Prints a summary
def AR_model(log_returns, tickers_with_AR):
    print("AR_model() start execute")
    tickers = tickers_with_AR
    #log_returns = to_log_return(data)
    summary = {}
    for ticker in tickers:
        print("AR_model() start execute in ticker: " + ticker)
        log_rtn = log_returns[ticker].dropna()
        model = AR(log_rtn)
        best_order = model.select_order(maxlag=LAGS, ic='aic')
        result = model.fit(best_order)
        aic = result.aic
        summary[ticker] = [best_order, aic, 'AR']
    with open('results/AR_models.txt', 'w') as results_file:
        for k in summary.keys():
            message = "Ticker: {} \t\t\tBest order: {} \tAIC = {}\n".format(k, summary[k][0], summary[k][1])
            results_file.write(message)
    return summary


# %% For tickers that don't feature and autocorrelation, we fit an MA model.
def MA_model(log_returns, tickers_with_AR):
    print("MA_model() start execute")
    tickers = tickers_with_AR
    #log_returns = to_log_return(data)
    summary = {}
    for ticker in tickers:
        print("MA_model() start execute in ticker: " + ticker)
        log_rtn = log_returns[ticker].dropna()
        lowest_aic = inf
        for order in range(1,LAGS):
            print("MA_model() start execute in order: " + str(order))
            model = ARMA(log_rtn, order=(0,order))
            result = model.fit()
            # result.summary()
            aic = result.aic
            if aic < lowest_aic:
                lowest_aic = aic
                best_order = order
                summary_best_fit = result.summary()
        summary[ticker] = [best_order, lowest_aic, 'MA']
        with open('results/MA_models.txt', 'a') as results_file:
            message = "\n\n\n\t\t\t\t\t********** {} **********\n \
                Best fit for {} obtained with model MA({})\n{}\n" \
                .format(ticker.upper(), ticker.upper(), best_order, summary_best_fit)
            results_file.write(message)
    return summary


# %% ARIMA modelling
def ARIMA_model(log_returns, tickers_with_AR):
    print("ARIMA_model() start execute")
    tickers = tickers_with_AR
    #log_returns = to_log_return(data)
    summary = {}
    for ticker in tickers:
        print("ARIMA_model() start execute in ticker: " + ticker)
        # AIC_table = {}
        # largest_pValue = {}
        # loglikelihood = {}
        order_error = []
        lowest_aic = inf
        log_rtn = log_returns[ticker].dropna()
        arma = [(1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
        arima = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
                 (1, 1, 0), (2, 1, 0), (0, 1, 1), (0, 1, 2)]
        orders = arma + arima
        for order in orders:
            try:
                model = ARIMA(log_rtn, order)
                result = model.fit()
                aic = result.aic
                if aic < lowest_aic:
                    lowest_aic = aic
                    best_order = order
                    summary_best_fit = result.summary()
            except:
                order_error.append(order)
        with open('results/ARMA_ARIMA_models.txt', 'a') as results_file:
            if best_order[1] == 0:
                best_model = 'ARMA'
            else:
                best_model = 'ARIMA'
            message = "\n\n\n\t********** {} **********\n \
Lowest AIC for {} obtained with model {}({})\n \
Please note the following orders returned errors: {}\n \
{}\n".format(ticker.upper(), \
             ticker.upper(), best_model, best_order, \
             order_error, \
             summary_best_fit)
            results_file.write(message)
        summary[ticker] = [best_order, lowest_aic, best_model]
    return summary


# %% SARIMA modelling
def SARIMA_model(data, tickers_with_AR):
    print("SARIMA_model() start execute")
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    mthly_log_returns = log_returns.resample('M').agg(lambda x: x[-1])
    summary = {}
    for ticker in tickers:
        print("SARIMA_model() start execute in ticker: " + ticker)
        order_error = []
        lowest_aic = inf
        log_rtn = mthly_log_returns[ticker].dropna()
        arma = [(1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
        arima = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
                 (1, 1, 0), (2, 1, 0), (0, 1, 1), (0, 1, 2)]
        orders = arma + arima
        seasonal_orders = [(1, 0, 1, 12), (2, 0, 1, 12), (1, 0, 2, 12), (2, 0, 2, 12)]
        for order in orders:
            try:
                for seasonal_order in seasonal_orders:
                    model = SARIMAX(log_rtn, order=order, seasonal_order=seasonal_order)
                    result = model.fit()
                    aic = result.aic
                    if aic < lowest_aic:
                        lowest_aic = aic
                        best_order = order
                        best_seasonal_order = seasonal_order
                        summary_best_fit = result.summary()
                        # with open('results/testing_SARIMA.txt', 'a') as f:
                        #     mess = '\n' + str(order) + ' ' + str(seasonal_order) + '\n' + str(summary_best_fit) + '\n'
                        #     f.write(mess)
            except:
                order_error.append([order, seasonal_order])
        with open('results/SARIMA_models.txt', 'a') as results_file:
            message = "\n\n\n\t********** {} **********\n \
Lowest AIC for {} obtained with SARIMA({}, {})\n \
Please note the following orders returned errors: {}\n \
{}\n".format(ticker.upper(), \
             ticker.upper(), best_order, best_seasonal_order, \
             order_error, \
             summary_best_fit)
            results_file.write(message)
        summary[ticker] = [best_order, lowest_aic]
    with open('results/SARIMA_models.txt', 'a') as results_file:
        results_file.write(str(summary))
    return summary


# %% ARIMAX modelling
def ARIMAX_model(data, tickers_with_AR):
    print("ARIMAX_model() start execute")
    summary = {}
    for ticker in tickers_with_AR:
        print("ARIMAX_model() start execute in ticker: " + ticker)
        log_returns = to_log_return(data)
        log_returns[ticker] = log_returns[ticker].shift(1)
        log_returns.dropna(how='any', inplace=True)
        endog = log_returns[ticker]
        exog = log_returns.drop(columns=ticker)
        order_error = []
        lowest_aic = inf
        arma = [(1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]
        arima = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2),
            (1, 1, 0), (2, 1, 0), (0, 1, 1), (0, 1, 2)]
        orders = arma + arima
        for order in orders:
            try:
                model = ARIMA(endog, order, exog=exog)
                result = model.fit()
                aic = result.aic
                if aic < lowest_aic:
                    lowest_aic = aic
                    best_order = order
                    summary_best_fit = result.summary()
            except:
                order_error.append(order)
        with open('results/ARMAX_ARIMAX_models.txt', 'a') as results_file:
            if best_order[1] == 0:
                best_model = 'ARMAX'
            else:
                best_model = 'ARIMAX'
            message = "\n\n\n\t********** {} **********\n \
Lowest AIC for {} obtained with model {}({})\n \
Please note the following orders returned errors: {}\n \
{}\n".format(ticker.upper(), \
            ticker.upper(), best_model, best_order, \
            order_error, \
            summary_best_fit)
            results_file.write(message)
        summary[ticker] = [best_order, lowest_aic, best_model]
    return summary


# # %% SARIMAX modelling
# def SARIMAX_model(data, tickers_with_AR):
#     print("SARIMAX_model() start execute")
#     tickers = tickers_with_AR
#     log_returns = to_log_return(data)
#     mthly_log_returns = log_returns.resample('M').agg(lambda x: x[-1])
#     for ticker in tickers:
#         print("SARIMAX_model() start execute in ticker: " + ticker)
#         log_rtn = mthly_log_returns[ticker].dropna()


def linear_model_fit_resid(log_rtn, ticker, linear_model_summary):
    if linear_model_summary[ticker][2] == 'AR':
        result = AR(log_rtn).fit(linear_model_summary[ticker][0])
    elif linear_model_summary[ticker][2] == 'MA':
        result = ARMA(log_rtn, order=(0, linear_model_summary[ticker][0])).fit()
    else:
        result = ARIMA(log_rtn, order=linear_model_summary[ticker][0]).fit()
    residual = result.resid
    return residual

# %% Tests for ARCH effect (serial autocorrelation of the volatility) in the log-returns
# signal=True   tickers are white noise data
# signal=False  tickers are model residual
def is_fit_for_ARCH(log_returns, tickers, summary, signal):
    print("is_fit_for_ARCH() start execute")
    #tickers = get_tickers(data)
    #tickers = get_tickers(log_returns)
    #log_returns = to_log_return(data)
    not_white_noise = []
    for ticker in tickers:
        print("is_fit_for_ARCH() start execute in ticker: " + ticker)
        log_rtn = log_returns[ticker].dropna()
        if signal:      # tickers are white noise data
            at2 = ((log_rtn - log_rtn.mean()) ** 2).dropna()
            if not is_white_noise(at2, nlags=LAGS, thres=0.05):
                not_white_noise.append(ticker)
        else:                                   # tickers are model residual
            residual = linear_model_fit_resid(log_rtn, ticker, summary)
            at2 = ((residual - residual.mean()) ** 2).dropna()
            if not is_white_noise(at2, nlags=LAGS, thres=0.05):
                not_white_noise.append(ticker)
    if signal:
        print("The following tickers are white noise series, \
              and are fit for ARCH model:", not_white_noise)
    else:
        print("The following tickers' model residual has ARCH effect (not white noise) ,\
              and are fit for ARCH model:", not_white_noise)
    return not_white_noise


def volatility_model(log_rtn, ticker, vol, p, q):
    print("volatility_model() start execute")
    if ticker == 'vix':
        log_rtn = log_rtn * 100  # rescaling as advised by optimizer
    else:
        log_rtn = log_rtn * 1000  # rescaling as advised by optimizer
    am = arch_model(log_rtn, vol=vol, p=p, q=q, dist='Normal')
    res = am.fit()
    # print(res.summary())
    return res


def resid_arch_fitting(log_returns, ticker, summary_model):
    print("arch_fitting() start execute")
    log_rtn = log_returns[ticker].dropna()
    residual = linear_model_fit_resid(log_rtn, ticker, summary_model)
    results_table = []
    lowest_aic = inf
    for vol in ['arch', 'garch', 'egarch']:
        for p in range(1, 3):
            for q in range(3):
                res = volatility_model(residual, ticker, vol, p, q)
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
    with open('results/'+summary_model[ticker][2]+'_ARCH_models.txt', 'a') as f:
        f.write(message)
    # input('Press <enter> to continue')
    return best_fit


def arch_fitting(log_returns, ticker):
    print("arch_fitting() start execute")
    log_rtn = log_returns[ticker].dropna()
    results_table = []
    lowest_aic = inf
    for vol in ['arch', 'garch', 'egarch']:
        for p in range(1, 3):
            for q in range(3):
                res = volatility_model(log_rtn, ticker, vol, p, q)
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
    return best_fit


# %%


# %%
def visualization(origin, prediction, model, ticker):
    print("visualization() start execute")
    prediction.plot(color='blue', label='Predict')
    origin.plot(color='red', label='Original')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % sqrt(sum((prediction - origin).dropna() ** 2) / origin.size))
    filename = './prediction_figure/' + model + '_for_' + ticker
    plt.savefig(filename)
    plt.clf()
    #    plt.show()
    return


def AR_prediction(data, test_data, test_for_AR, ar_summary):
    print("AR_prediction() start execute")
    tickers = test_for_AR
    log_returns = to_log_return(data)
    with open('prediction_results/AR_model_prediction.txt', 'w') as results_file:
        for ticker in tickers:
            print("AR_prediction() start execute in ticker: " + ticker)
            log_rtn = log_returns[ticker].dropna()
            result = AR(log_rtn).fit(ar_summary[ticker][0])
            result_show = result.predict(test_data.index[0], test_data.index[-1])
            test_log_returns = to_log_return(test_data)
            test_log_rtn = test_log_returns[ticker].dropna()
            test_log_rtn = test_log_rtn[result_show.index]
            visualization(test_log_rtn, result_show, 'AR_prediction', ticker)
            rmse = sqrt(sum((result_show - test_log_rtn).dropna() ** 2) / test_log_rtn.size)
            message = "The prediction for {} is \n {}\n RMSE:{}\n".format(ticker, result_show, rmse)
            results_file.write(message)

    return


def MA_prediction(data, test_data, tickers_without_AR, summary_MA):
    print("MA_prediction() start execute")
    tickers = tickers_without_AR
    log_returns = to_log_return(data)
    with open('prediction_results/MA_model_prediction.txt', 'w') as results_file:
        for ticker in tickers:
            print("MA_prediction() start execute in ticker: " + ticker)
            log_rtn = log_returns[ticker].dropna()
            model = ARMA(log_rtn, order=(0, summary_MA[ticker][0]))
            result = model.fit()
            result_show = result.predict(test_data.index[0], test_data.index[-1])
            test_log_returns = to_log_return(test_data)
            test_log_rtn = test_log_returns[ticker].dropna()
            test_log_rtn = test_log_rtn[result_show.index]
            visualization(test_log_rtn, result_show, 'MA_prediction', ticker)
            rmse = sqrt(sum((result_show - test_log_rtn).dropna() ** 2) / test_log_rtn.size)
            message = "The prediction for {} is \n {}\n RMSE:{}\n".format(ticker, result_show, rmse)
            results_file.write(message)
    return


def ARIMA_prediction(data, test_data, tickers_with_AR, summary_ARIMA):
    print("ARIMA_prediction() start execute")
    tickers = tickers_with_AR
    log_returns = to_log_return(data)
    with open('prediction_results/ARIMA_model_prediction.txt', 'w') as results_file:
        for ticker in tickers:
            print("ARIMA_prediction() start execute in ticker: " + ticker)
            log_rtn = log_returns[ticker].dropna()
            model = ARIMA(log_rtn, order=summary_ARIMA[ticker][0])
            result = model.fit()
            result_show = result.predict(test_data.index[0], test_data.index[-1])
            test_log_returns = to_log_return(test_data)
            test_log_rtn = test_log_returns[ticker].dropna()
            test_log_rtn = test_log_rtn[result_show.index]
            visualization(test_log_rtn, result_show, 'ARIMA_prediction', ticker)
            rmse = sqrt(sum((result_show - test_log_rtn).dropna() ** 2) / test_log_rtn.size)
            message = "The prediction for {} is \n {}\n RMSE:{}\n".format(ticker, result_show, rmse)
            results_file.write(message)
    return


def ARCH_prediction(data, test_data, tickers, summary_ARCH):
    print("ARCH_prediction() start execute")
    tickers = tickers
    log_returns = to_log_return(data)
    test_log_returns = to_log_return(test_data)
    with open('prediction_results/ARCH_model_prediction.txt', 'w') as results_file:
        for ticker in tickers:
            print("ARCH_prediction() start execute in ticker: " + ticker)
            log_rtn = log_returns[ticker].dropna()
            model = arch_model(log_rtn, vol=summary_ARCH[ticker][0], p=summary_ARCH[ticker][1][0],
                               q=summary_ARCH[ticker][1][1], dist='Normal')
            result = model.fit(last_obs='2019-9-23', update_freq=5)
            result_show = result.forecast()
            test_log_rtn = test_log_returns[ticker].dropna()
            print(result_show.mean.dropna().head())
            print(result_show.variance.dropna().head())

            mean = result_show.mean['h.1'].dropna()
            variance = result_show.variance['h.1'].dropna()
            line1 = mean + 1.96 * variance
            line2 = mean - 1.96 * variance

            line1.plot(color='blue', label='Predict1')
            line2.plot(color='blue', label='Predict2')
            test_log_rtn.plot(color='red', label='Original')

            plt.legend(loc='best')
            rmse = sqrt(sum((mean - test_log_rtn).dropna() ** 2) / test_log_rtn.size)
            plt.title('RMSE: %.4f' % rmse)
            filename = './prediction_figure/' + 'Garch_model' + '_for_' + ticker
            plt.savefig(filename)
            plt.clf()
            #            plt.show()
            message = "The prediction mean for {} is \n {}\n The prediction variance is\n{}\n RMSE:{}\n ".format(ticker,
                                                                                                                 mean,
                                                                                                                 variance,
                                                                                                                 rmse)
            results_file.write(message)
    return
