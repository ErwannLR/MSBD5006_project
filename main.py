#%%
from STATICS import FILE
import functions as f

# Prepare workspace
f.cleanup()

# Load/prepare data
data = f.load_data(FILE)

#%% graph multicharts and save to folder
f.generate_multicharts(data)

#%% find which tickers are not white noise, 
# # i.e. demonstrate some level of autocorrelation in the log-returns
tickers_with_AR, tickers_without_AR = f.is_fit_for_AR(data)

#%% Perform AR Model fitting for tickers with autocorrelation in the returns
f.AR_model(data, tickers_with_AR)

# %% For remaining tickers, we test MA models.
f.MA_model(data, tickers_without_AR)

#%% For tickers featuring autocorrelation of the returns, we now test for ARMA and ARIMA
f.ARIMA_model(data, tickers_with_AR)

#%% find which tickers feature volatility that is relevant 
# for ARCH modelling, i.e. where there is serial autocorrelation 
# in the volatility of the log-returns
test_for_ARCH = f.is_fit_for_ARCH(data)
# print("The following tickers are relevant for ARCH modeling:", test_for_ARCH)

# Perform ARCH fitting for relevant tickers
for ticker in test_for_ARCH:
    f.arch_fitting(data, ticker)
# %%
 