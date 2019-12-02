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


# %%
