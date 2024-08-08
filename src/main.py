from data_processing import (
    read_cyclone_file, process_cyclone_data, concatenate_cyclone_data,
    filter_and_process_data, analyze_hurricanes, process_amo_data,
    merge_hurricane_amo, impute_slp_data, process_slp_data
)
from modeling import (
    logistic_regression_and_visualization, random_forest_classifier
)

import pandas as pd
import joblib
"""As long as your file paths are good it should run with updated information"""

if __name__ == "__main__":

    # Update file path if with new data files
    file_path = 'data/hurdat2_1851_2023.txt'
    file_path1 = 'data/78073_Nassau_MSLP_18502004_v2.fts'
    file_path2 = 'data/72208_Charleston_MSLP_18502004_v2.fts'
    file_path3 = 'data/76644_Merida_MSLP_18502004_v2.fts'
    amo_file_path = 'data/amo_sst.txt'

    # These next values are for the process_slp_data function and amo_sst
    columns2 = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    drop_columns_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 
                                    'Jun', 'Jul', 'Nov', 'Dec']
    months = ['Aug', 'Sep', 'Oct']
    replace_value = -999.9

    # Cleaning the NHC hurdat file
    cyclone_data = read_cyclone_file(file_path)
    all_cyclone_dfs = process_cyclone_data(cyclone_data)
    all_data = concatenate_cyclone_data(all_cyclone_dfs)

    # Define the latitude and longitude for filtering the location
    min_lat, max_lat = 27.5, 29.4
    min_lon, max_lon = -81.5, -78.8

    filtered_data = filter_and_process_data(all_data, min_lat, max_lat, min_lon, max_lon)

    # Breaking down hurricanes specifically, hu_binary captures whether a hurricane happened in a year as 0 or 1
    hu_trend, hu_binary = analyze_hurricanes(filtered_data)

    # amo data is the temp anomalies in the North Atlantic
    amo_data = process_amo_data(amo_file_path, columns2)

    # Merge amo and hurricanes binary for the first model
    hu_binary_amo_df, hu_amo_df = merge_hurricane_amo(hu_binary, hu_trend, amo_data)

    # These next six are to capture sea level pressure at three locations, then the extra dataframes are to help impute na values
    meridia_mean = process_slp_data(file_path3, columns2, drop_columns_list, months, replace_value, 'Meridia_slp')
    charleston_mean = process_slp_data(file_path2, columns2, drop_columns_list, months, replace_value, 'Charleston_slp')
    nassau_mean = process_slp_data(file_path1, columns2, drop_columns_list, months, replace_value, 'Nassau_slp')
    tna_sst = pd.read_csv('data/TNA_sst.txt', 
                    delim_whitespace=True, 
                    index_col='Year')
    amm_sst = pd.read_csv('data/AMM_sst.txt', 
                    delim_whitespace=True, 
                    index_col='Year')
    rh_value = pd.read_csv('data/rh_mdr.txt', 
                    delim_whitespace=True, 
                    index_col='Year')
    
    additional_dfs = [nassau_mean, charleston_mean, meridia_mean, tna_sst, amm_sst, rh_value]

    imputed_slp, hu_slp_df = impute_slp_data(amo_data, additional_dfs, hu_binary)
    
    # logistic model for the initial model between amo and hu_binary
    logit_model = logistic_regression_and_visualization(hu_binary_amo_df)
    
    # This is for the second model using sea level pressure, amo, and hurricanes binary
    rf_classifier, accuracy, precision, recall, f1, conf_matrix = random_forest_classifier(hu_slp_df, amo_data)

    # Next we dump the new files into respective folders to be pulled into app.py
    joblib.dump(rf_classifier, 'models/rf_model.joblib')
    joblib.dump(logit_model, 'models/initial_regression_model.joblib')
    joblib.dump(hu_binary_amo_df, 'joblib_files/hu_binary_amo_df.joblib')
    joblib.dump(hu_slp_df, 'joblib_files/hurricane_slp_df.joblib')
    joblib.dump(hu_amo_df, 'joblib_files/hu_amo_original.joblib')

    # Print evaluation results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
