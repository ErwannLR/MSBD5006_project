import pandas as pd
import os
from matplotlib import pyplot

import ExploratoryAnalysis as ea

raw_data = pd.read_csv("closing_prices.csv")
# LEAVING THE STEPS I EXECUTED FOR OTHERS TO KNOW WHAT I DID
for col_name in raw_data:
    if col_name != ea.DATES_COL_NAME:
        transformed_data = ea.transform_and_save(raw_data, col_name, False,
                                                 False)  # remove NANs, add log returns and save for each commodity
        # ea.AFD_test(transformed_data[ea.LOG_RETURNS_COL_NAME], col_name).to_csv(
        #    ea.TRANSFORMED_DATA_LOCATION + "adf_results" + ea.CSV_EXT, mode='a')
        if ea.is_white_noise(transformed_data[ea.LOG_RETURNS_COL_NAME]):
            print(col_name, "is white noise")
        else:
            print(col_name, "is NOT white noise")

        #break
        # ea.acf_pacf_plot(transformed_data[ea.LOG_RETURNS_COL_NAME],col_name)

# Iterating over each of the files in the transformed_data directory
# for file in os.listdir(ea.TRANSFORMED_DATA_LOCATION):
#    data = pd.read_csv(ea.TRANSFORMED_DATA_LOCATION + file)
