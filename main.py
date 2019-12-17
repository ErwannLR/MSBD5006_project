# %%
from STATICS import FILE
import functions as f

try:
    # Prepare workspace
    # NB: uncomment only if you have time to re-run the whole code to reproduce results (several hours)
    # f.cleanup()

    # Load/prepare data
    data = f.load_data(FILE)

    # Convert data into log-reutrntic
    log_returns = f.to_log_return(data)

    # %% graph multicharts and save to folder
    # f.generate_multicharts(data)

    # %% find which tickers are not white noise,
    # # i.e. demonstrate some level of autocorrelation in the log-returns
    tickers_with_AR, tickers_without_AR = f.is_fit_for_AR(log_returns)


    # %% Perform AR Model fitting for tickers with autocorrelation in the returns
    summary_AR = f.AR_model(log_returns, tickers_with_AR)
    # %% Perform MA Model fitting for tickers featuring autocorrelation of the returns
    summary_MA = f.MA_model(log_returns, tickers_without_AR)
    # %% Perform ARMA and ARIMA fitting for tickers featuring autocorrelation of the returns
    summary_ARIMA = f.ARIMA_model(log_returns, tickers_with_AR)
    # %% ARIMAX Modelling
    summary_ARIMAX = f.ARIMAX_model(data, tickers_with_AR)
    %%  SARIMA Modelling
    summary_SARIMA = f.SARIMA_model(data, tickers_with_AR)

    # %% split data into two parts
    train_size = int(data.shape[0] - 10)
    test_data = data.iloc[train_size:, :]
    train_data = data.iloc[0:train_size, :]
    data = train_data

    # %% ARIMA Prediction
    f.AR_prediction(data, test_data, tickers_with_AR, summary_AR)
    f.MA_prediction(data, test_data, tickers_with_AR, summary_MA)
    f.ARIMA_prediction(data, test_data, tickers_with_AR, summary_ARIMA)


    # %% find which tickers feature volatility that is relevant
    # for ARCH modelling, i.e. where there is serial autocorrelation
    # in the volatility of the log-returns
    # AR model residual test for ARCH effect
    test_AR_for_ARCH = f.is_fit_for_ARCH(log_returns, tickers_with_AR, summary_AR, signal=False)
    # %% Perform ARCH fitting for AR model residual relevant tickers
    summary_AR_ARCH = {}
    for ticker in test_AR_for_ARCH:
        print("arch_fitting() start execute in ticker: " + ticker)
        best_fit = f.resid_arch_fitting(log_returns, ticker, summary_AR)
        summary_AR_ARCH[ticker] = [best_fit[1], best_fit[2], best_fit[3]]

    # MA model residual test for ARCH effect
    test_MA_for_ARCH = f.is_fit_for_ARCH(log_returns, tickers_with_AR, summary_MA, signal=False)
    # %% Perform ARCH fitting for MA model residual relevant tickers
    summary_MA_ARCH = {}
    for ticker in test_MA_for_ARCH:
        print("arch_fitting() start execute in ticker: " + ticker)
        best_fit = f.resid_arch_fitting(log_returns, ticker, summary_MA)
        summary_MA_ARCH[ticker] = [best_fit[1], best_fit[2], best_fit[3]]

    # ARMA or ARIMA model residual test for ARCH effect
    test_ARIMA_for_ARCH = f.is_fit_for_ARCH(log_returns, tickers_with_AR, summary_ARIMA, signal=False)
    # %% Perform ARCH fitting for ARMA or ARIMA model residual relevant tickers
    summary_ARIMA_ARCH = {}
    for ticker in test_ARIMA_for_ARCH:
        print("arch_fitting() start execute in ticker: " + ticker)
        best_fit = f.resid_arch_fitting(log_returns, ticker, summary_ARIMA)
        summary_ARIMA_ARCH[ticker] = [best_fit[1], best_fit[2], best_fit[3]]

    # data with white nosie test for ARCH effect
    test_for_ARCH = f.is_fit_for_ARCH(log_returns, tickers_without_AR, 0, signal=True)
    # %% Perform ARCH fitting for relevant tickers
    summary_ARCH = {}
    for ticker in test_for_ARCH:
        print("arch_fitting() start execute in ticker: " + ticker)
        best_fit = f.arch_fitting(log_returns, ticker)
        summary_ARCH[ticker] = [best_fit[1], best_fit[2], best_fit[3]]



    # %% ARCH prediction
    data = f.load_data(FILE)
    f.ARCH_prediction(data, test_data, test_for_ARCH, summary_ARCH)

    print("Algorithm executed without exception")

except Exception as e:
    print(f"Exception is thrown:" + str(e))
