import pandas as pd
import numpy as np
from statsmodels.stats import diagnostic
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

CHARTS_LOCATION = "charts\\"
TRANSFORMED_DATA_LOCATION = "transformed_data\\"
LOG_RETURNS_COL_NAME = "log_returns"
DATES_COL_NAME = "Dates"
CSV_EXT = ".csv"
PNG_EXT = ".png"

LAGS = 300

ALPHA = 0.05


def delete_nans(data, col_name):
    return data[col_name].dropna()


def plot_decomposed_ts(series, name):
    # https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
    # Currently, 258 = total number of rows / total number of years = 7766 / 30
    # we can improve by finding the number of business days in a year for each commodity?
    result = seasonal_decompose(series, model='additive', freq=258)  # TODO: calculate proper freq?
    result.plot()
    pyplot.savefig(CHARTS_LOCATION + (name + PNG_EXT).replace(" ", "_").lower())


def transform_and_save(data, col, plot, save):
    t_data = pd.concat([data[DATES_COL_NAME], data[col]], axis=1, keys=[DATES_COL_NAME, col])
    t_data = t_data.dropna()

    # t_data = t_data.set_index(pd.DatetimeIndex(t_data[DATES_COL_NAME])).drop(DATES_COL_NAME)
    if plot:
        # pd.Series(t_data[col], index=pd.DatetimeIndex(t_data[DATES_COL_NAME]))
        ts = t_data
        ts[DATES_COL_NAME] = pd.to_datetime(ts[DATES_COL_NAME])
        ts = t_data.set_index(pd.DatetimeIndex(t_data[DATES_COL_NAME]))
        ts = ts.drop(columns=[DATES_COL_NAME])
        plot_decomposed_ts(ts, col + "_decomposed")
        plot_time_series(t_data[DATES_COL_NAME], t_data[col], col)

    t_data[LOG_RETURNS_COL_NAME] = log_returns(t_data, col)
    t_data = t_data.dropna()
    if plot:
        plot_time_series(t_data[DATES_COL_NAME], t_data[LOG_RETURNS_COL_NAME], col + "_" + LOG_RETURNS_COL_NAME)

    if save:
        t_data.to_csv((TRANSFORMED_DATA_LOCATION + col + CSV_EXT).replace(" ", "_").lower())

    return t_data


def is_white_noise(col, lags=LAGS, box_pierce=False):
    # https://stats.stackexchange.com/questions/200267/interpreting-ljung-box-test-results-from-statsmodels-stats-diagnostic-acorr-lju
    ljung_box_result, pvals = diagnostic.acorr_ljungbox(col, lags, box_pierce)
    for val in pvals:
        if val > ALPHA:
            return True
    return False


def log_returns(df, col_name):
    ratio = df[col_name] / df[col_name].shift(1)
    return np.log(ratio)


def plot_time_series(x_col, y_col, figure_name):
    location = CHARTS_LOCATION + (figure_name + PNG_EXT).replace(" ", "_").lower()
    print("Plotting " + location)
    df = pd.DataFrame()
    df["date"] = x_col
    df["prices"] = y_col
    plot = df.plot("date", "prices")
    plot.set_title(figure_name)
    plot.figure.savefig(location)


def acf_pacf_plot(col, name):
    # https://blogs.oracle.com/datascience/performing-a-time-series-analysis-on-the-sandp-500-stock-index
    #lags
    plot = plot_acf(col, lags=LAGS, marker=None)
    pyplot.savefig(CHARTS_LOCATION + (name + "_acf" + PNG_EXT).replace(" ", "_").lower())


def ADF_test(col, name):
    # https://machinelearningmastery.com/time-series-data-stationary-python/
    # lags?
    result = adfuller(col)
    df = pd.DataFrame()
    df["instrument"] = [name]
    df["ADF Statistic"] = [result[0]]
    df["p-value"] = [result[1]]
    for key, value in result[4].items():
        df["Critical Value " + str(key)] = [value]
    df.set_index("instrument")
    return df
