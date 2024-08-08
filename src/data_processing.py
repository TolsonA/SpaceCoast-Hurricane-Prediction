import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# This is where we start the data cleaning. First functions are for the NHC hurdat file.

def convert_lat_lon(value):
    """Convert latitude and longitude values to float."""
    if 'N' in value or 'E' in value:
        return float(value[:-1])
    elif 'S' in value or 'W' in value:
        return -float(value[:-1])

def read_cyclone_file(file_path):
    """Read cyclone data from a file and organize it into a list of cyclones."""
    cyclone_data = []
    current_cyclone = None

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('AL'):
                if current_cyclone is not None:
                    cyclone_data.append(current_cyclone)
                current_cyclone = {'header': line.strip(), 'data': []}
            else:
                if current_cyclone is not None:
                    current_cyclone['data'].append(line.strip().split(','))

    if current_cyclone is not None:
        cyclone_data.append(current_cyclone)
    
    return cyclone_data

def process_cyclone_data(cyclone_data):
    """Process cyclone data into a list of DataFrames."""
    all_cyclone_dfs = []

    for cyclone in cyclone_data:
        df = pd.DataFrame(cyclone['data'], columns=['Date', 'Time', 'Record', 'Status', 'Latitude', 'Longitude', 'WindSpeed', 'Pressure',
                                                    'Rad_34_NE', 'Rad_34_SE', 'Rad_34_SW', 'Rad_34_NW', 'Rad_50_NE', 'Rad_50_SE',
                                                    'Rad_50_SW', 'Rad_50_NW', 'Rad_64_NE', 'Rad_64_SE', 'Rad_64_SW', 'Rad_64_NW', 'maxwnd'])
        # Convert data types
        df['Date'] = df['Date'].astype(str)
        df['Time'] = df['Time'].astype(str)
        df['Latitude'] = df['Latitude'].apply(convert_lat_lon)
        df['Longitude'] = df['Longitude'].apply(convert_lat_lon)
        df['WindSpeed'] = df['WindSpeed'].astype(int)
        df['Pressure'] = df['Pressure'].astype(int)
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y%m%d %H%M')
        df.set_index('Datetime', inplace=True)
        df['Cyclone'] = cyclone['header']  # Add a column for the cyclone identifier
        all_cyclone_dfs.append(df)
    
    return all_cyclone_dfs

def concatenate_cyclone_data(all_cyclone_dfs):
    """Concatenate all cyclone DataFrames into a single DataFrame."""
    return pd.concat(all_cyclone_dfs)

def filter_and_process_data(df, min_lat, max_lat, min_lon, max_lon):
    """Filter data by lat/lon, assign status priority, and extract year."""
    
    df = df[(df['Latitude'] >= min_lat) & (df['Latitude'] <= max_lat) &
                       (df['Longitude'] >= min_lon) & (df['Longitude'] <= max_lon)]
    
    filtered_data = df.copy()
    
    priority_order = {'HU': 1, 'TS': 2, 'TD': 3}
    filtered_data['StatusPriority'] = filtered_data['Status'].map(priority_order).fillna(4)
    
    # Extract the year from the Datetime index
    filtered_data['Year'] = filtered_data.index.year
    
    return filtered_data

def analyze_hurricanes(filtered_data):
    """Analyze hurricane data to get trends and binary indicators."""
    only_hurricane = filtered_data.copy()
    only_hurricane['Status'] = only_hurricane['Status'].astype(str).str.strip()
    only_hurricane = only_hurricane[only_hurricane['Status'] == 'HU']

    # Calculate hurricane trends
    hu_trend = only_hurricane.groupby('Year')['Cyclone'].nunique()
    all_years = pd.Series(0, index=np.arange(filtered_data['Year'].min(), 
                                             filtered_data['Year'].max() + 1))
    hu_trend = hu_trend.reindex(all_years.index, fill_value=0)
    hu_trend = pd.DataFrame(hu_trend)
    hu_trend.rename(columns={'Cyclone': 'Hurricanes'}, inplace=True)
    hu_trend.index.name = 'Year'

    # Create binary hurricane data
    hu_binary = hu_trend.copy()
    hu_binary['Hurricanes'] = hu_binary['Hurricanes'].apply(lambda x: 1 if x > 0 else 0)
    
    return hu_trend, hu_binary

def process_amo_data(file_path, columns):
    """Process AMO data from a file."""
    amo_data = pd.read_csv(file_path, delim_whitespace=True, names=columns)
    amo_data = amo_data.drop(columns=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Nov', 'Dec'])
    amo_data['Annual'] = np.mean(amo_data.loc[:, 'Jun':'Oct'].values, axis=1)
    amo_data = amo_data[['Year', 'Annual']]
    amo_data.rename(columns={'Annual': 'AMO_Anomaly'}, inplace=True)
    amo_data.set_index('Year', inplace=True)
    return amo_data

def merge_hurricane_amo(hu_binary, hu_trend, amo_data):
    """Merge hurricane data with AMO data."""
    hu_binary_amo_df = hu_binary.merge(amo_data, on='Year')
    hu_amo_df = hu_trend.merge(amo_data, on='Year')
    return hu_binary_amo_df, hu_amo_df

def process_slp_data(file_path, columns, drop_columns, months, replace_value, rename_column):
    """Process the data to prepare for merge."""
    data = pd.read_csv(file_path, delim_whitespace=True, names=columns)
    data = data[1:].drop(columns=drop_columns)
    
    # Convert specified columns to float
    data[months[-2:]] = data[months[-2:]].astype(float)
    
    # Replace -999.9 values and forward fill missing data
    data.replace(replace_value, pd.NA, inplace=True)
    data.ffill(inplace=True)
    
    # Calculate annual values
    data['Annual'] = data[months].mean(axis=1) - data[months].mean(axis=1).mean()
    
    # Prepare for merging
    data_copy = data.copy()
    data_copy['Year'] = data_copy['Year'].astype(int)
    data_copy = data_copy[['Year', 'Annual']]
    data_copy.set_index('Year', inplace=True)
    data_copy.rename(columns={'Annual': rename_column}, inplace=True)
    
    return data_copy


def impute_slp_data(amo_data, additional_dfs, hu_binary):
    """Join and impute SLP data, and combine with hurricane data."""
    slp_df = amo_data.join(additional_dfs, how='left')
    df_imputer = slp_df[94:]
    mice_imputer = IterativeImputer(random_state=42)
    df_imputer = pd.DataFrame(mice_imputer.fit_transform(df_imputer), columns=df_imputer.columns)
    df_imputer = pd.DataFrame(mice_imputer.transform(df_imputer), columns=df_imputer.columns)
    df_imputer = df_imputer[['Nassau_slp', 'Charleston_slp', 'Meridia_slp']]
    imputed_slp = df_imputer[-18:]
    imputed_slp = imputed_slp.set_index(slp_df.index[-18:])

    # Adding the imputed numbers to the sea level pressure stations
    only_slp_stations = additional_dfs[0].join([additional_dfs[1], additional_dfs[2]], how='left')
    only_slp_stations = pd.concat([only_slp_stations, imputed_slp])

    # Combining the SLP stations and binary hurricanes for the second model
    hu_slp_df = only_slp_stations.merge(hu_binary, on='Year')
    
    return imputed_slp, hu_slp_df


