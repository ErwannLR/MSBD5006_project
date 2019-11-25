#%%
from STATICS import FILE
import functions as f

# Load/prepare data
data = f.load_data(FILE)
# graph multicharts and save to folder
f.generate_multicharts(data)
# find which tickers are not white noise, i.e. demonstrate some level of autocorrelation in the log-returns
test_for_AR = f.fit_for_AR(data)
# perform AR Model fitting for relevant ttickers

