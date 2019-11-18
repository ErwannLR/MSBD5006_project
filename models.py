#%% Imports
from statsmodels.tsa.ar_model import AR
from STATICS import LAGS
# from functions import get_tickers, to_log_return

def AR_model(log_returns):
    for log_rtn in log_returns
        model = AR(log_rtn)
        best_order = model.select_order(maxlag=LAGS, ic='aic')
        model.fit(best_order)