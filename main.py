#%%
from STATICS import FILE
import functions as f

try:
    # Prepare workspace
    f.cleanup()

    # Load/prepare data
    data =f.load_data(FILE)

    #%% graph multicharts and save to folder
    f.generate_multicharts(data)

    #%% find which tickers are not white noise,
    # # i.e. demonstrate some level of autocorrelation in the log-returns
    tickers_with_AR, tickers_without_AR = f.is_fit_for_AR(data)

    #%% split data into two parts
    train_size = int(data.shape[0]-10)
    test_data = data.iloc[train_size:,:]
    train_data = data.iloc[0:train_size,:]
    data = train_data

    #%% Perform AR Model fitting for tickers with autocorrelation in the returns
    summary_AR = f.AR_model(data, tickers_with_AR)

    # %% For remaining tickers, we test MA models.
    summary_MA = f.MA_model(data, tickers_without_AR)

    #%% For tickers featuring autocorrelation of the returns, we now test for ARMA and ARIMA
    summary_ARIMA = f.ARIMA_model(data, tickers_with_AR)

    #%% ARIMA Prediction
    f.AR_prediction(data, test_data, tickers_with_AR, summary_AR)
    f.MA_prediction(data, test_data, tickers_without_AR, summary_MA)
    f.ARIMA_prediction(data, test_data, tickers_with_AR, summary_ARIMA)

    #%% find which tickers feature volatility that is relevant 
    # for ARCH modelling, i.e. where there is serial autocorrelation 
    # in the volatility of the log-returns
    test_for_ARCH = f.is_fit_for_ARCH(data)
    print("The following tickers are relevant for ARCH modeling:", test_for_ARCH)

    #%% Perform ARCH fitting for relevant tickers
    summary_ARCH = {}
    for ticker in test_for_ARCH:
        print("arch_fitting() start execute in ticker: " + ticker)
        best_fit = f.arch_fitting(data, ticker)
        summary_ARCH[ticker] = [best_fit[1],best_fit[2],best_fit[3]]

    #%% ARCH prediction
    data =f.load_data(FILE)
    f.ARCH_prediction(data,test_data,test_for_ARCH, summary_ARCH)

    print("Algorithm executed without exception")
except Exception as e:
    print(f"Exception is thrown:" + str(e))
    