#%% Imports
from statsmodels.tsa.ar_model import AR
from STATICS import LAGS
# from functions import get_tickers, to_log_return

def AR_model(log_returns):
    for log_rtn in log_returns:
        model = AR(log_rtn)
        best_order = model.select_order(maxlag=LAGS, ic='aic')
        model.fit(best_order)
    return

# *** WORK IN PROGRESS ***
#%% Imports for ACH modelisation
#%% ARCH modelisation
# Note all tests use the 5% significance level for type-I error and use
# ten lags in all ACF or ARCH-effect tests.
lags = 10
signif_level = 0.05
#%% Load, process and plot data
cols = ['date', 'sbux', 'snp']
df = pd.read_csv(file, names=cols, delim_whitespace=True)

# r = log(R + 1)
df['sbux_log'] = log(df.sbux + 1)
df['snp_log'] = log(df.snp + 1)
df = df[['date', 'sbux_log']]

plt.plot(df.sbux_log)

#%% Is there any serial correlation in the log returns of Starbucks stock?
plot_pacf(df.sbux_log, lags=lags, title="Starbucks log-returns PACF")
plot_acf(df.sbux_log, lags=lags, title="Starbucks log-returns ACF")

# %% ACF, q-stat, p-values
# H0: The data are independently distributed (i.e. the correlations in \ 
# the population from which the sample is taken are 0, so that any observed correlations in the data result from randomness of the sampling process).
# Ha: The data are not independently distributed; they exhibit serial correlation.
# => if p-values all below confidence level (here, 5%), then we reject H0, i.e. this is not white noise
a = acf(df.sbux_log, nlags=lags, fft=False, qstat=True)
acf_coeff = a[0]
q_stat = a[1]
p_values = a[2]
print('ACF coeff\n', acf_coeff)
print('q-stat\n', q_stat)
print('p-values\n', p_values)
autocorrel = False
for p in p_values:
    if p < signif_level:
        autocorrel = True
if autocorrel:
    print("There IS serial autocorrelation in the log-returns")
else:
    print("There is NO serial autocorrelation in the log-returns")

#%% Testing for ARCH effect
at2 = (df.sbux_log - mean(df.sbux_log)) ** 2
plot_acf(at2, lags=lags)
acf(at2, nlags=lags, fft=False, qstat=True)
# Yes there seems to be an ARCH effect
#%% Is there any serial correlation in the log returns of Starbucks stock?
plot_pacf(at2, lags=lags, title="Starbucks squared series of innovations PACF")
plot_acf(at2, lags=lags, title="Starbucks squared series of innovations ACF")

# %% ACF, q-stat, p-values
# H0: The data are independently distributed (i.e. the correlations in \ 
# the population from which the sample is taken are 0, so that any observed correlations in the data result from randomness of the sampling process).
# Ha: The data are not independently distributed; they exhibit serial correlation.
# => if p-values all below confidence level (here, 5%), then we reject H0, i.e. this is not white noise
a = acf(at2, nlags=lags, fft=False, qstat=True)
acf_coeff = a[0]
q_stat = a[1]
p_values = a[2]
print('ACF coeff\n', acf_coeff)
print('q-stat\n', q_stat)
print('p-values\n', p_values)
autocorrel = False
for p in p_values:
    if p < signif_level:
        autocorrel = True
if autocorrel:
    print("There IS ARCH effect (serial autocorrelation of the volatility) in the log-returns")
else:
    print("There is NO ARCH effect (serial autocorrelation) in the log-returns")


#%% Re-scaling data
data = 100 * df.sbux_log
# %% Fitting GARCH(1, 1)
am = arch_model(data, vol='Garch', p=1, q=1, dist='Normal')
res = am.fit()
print(res.summary())

# %% Model Checking
st_resid = res.resid / res.conditional_volatility
is_white_noise = False
for pvalue in LB(st_resid, lags=lags)[1]:
    if pvalue >= signif_level:
        is_white_noise = True
if is_white_noise:
    print("Standardized residuals are white noise, model is adequate")
else:
    print("Standardized residuals are not white noise, model is inadequate")

