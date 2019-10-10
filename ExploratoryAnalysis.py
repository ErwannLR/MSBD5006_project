import pandas as pd
import numpy as np
from statsmodels.stats import diagnostic

CHARTS_LOCATION = "charts\\"


def transform_data(data):
    # simply drop any rows with an empty cell, at least for now
    # we need to make sure that the data we end up with is consecutive(?). So we should pick the block where no rows have nan
    return data.dropna()


def ensure_stationarity(col, lags, box_pierce=False):
    # check if column is stationary, if not, convert it
    ljung_box_result = diagnostic.acorr_ljungbox(col, lags, box_pierce)
    print(ljung_box_result)


def log_returns(col):
    # the first log return will be nan
    pct_change = col.pct_change()
    return np.log(1 + pct_change)


def plot_time_series(x_col, y_col, figure_name):
    location = CHARTS_LOCATION + figure_name + ".png"
    print("Plotting " + location)
    df = pd.DataFrame()
    df["date"] = x_col
    df["prices"] = y_col
    plot = df.plot("date", "prices")
    plot.set_title(figure_name)
    plot.figure.savefig(location)
