import pandas as pd
import ExploratoryAnalysis as expAnalysis

data = pd.read_csv("closing_prices.csv")
data = expAnalysis.transform_data(data)
# data.to_csv("transformed.csv") # re-do once expAnalysis.transform_data is fixed and repeat the below

for columnName in data.iloc[:, 1:]:
    data[columnName] = expAnalysis.log_returns(data[columnName])
    #    log_returns = expAnalysis.log_returns(data[columnName])
    #    expAnalysis.plot_time_series(data["Dates"], log_returns, columnName.replace(" ", "_") + "_returns_vs_time")

data = data.dropna()  # not using expAnalysis.transform_data as we should no simply drop rows there
expAnalysis.ensure_stationarity(data["XAU Curncy"], 1)
