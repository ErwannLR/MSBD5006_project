from sys import platform

system = platform.lower()
if system == 'windows':
    CHARTS_LOCATION = 'multicharts\\'
else:
    CHARTS_LOCATION = 'multicharts/'
# Input data file name
FILE = 'closing_prices.csv'
# Nb lags to run Autoregression
LAGS=20
