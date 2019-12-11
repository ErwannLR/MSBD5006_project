#%%
from STATICS import FILE
import functions as f


# Load/prepare data
data = f.load_data(FILE)
# graph multicharts and save to folder
f.generate_multicharts(data)
# find which tickers are not white noise, i.e. demonstrate some level of autocorrelation in the log-returns
test_for_AR = f.is_fit_for_AR(data)
print("The following tickers are not white noise, and are fit for AR model:", test_for_AR)
# perform AR Model fitting for relevant tickers
f.AR_model(data, test_for_AR)
# find which tickers feature volatility that is relevant for ARCH modelling, 
# i.e. where there is serial autocorrelation in the volatility of the log-returns
test_for_ARCH = f.is_fit_for_ARCH(data)
print("The following tickers are relevant for ARCH modeling:", test_for_ARCH)
#Perform ARCH fitting for relevant tickers
for ticker in test_for_ARCH:
    f.arch_fitting(data, ticker)


# %%


# %%
