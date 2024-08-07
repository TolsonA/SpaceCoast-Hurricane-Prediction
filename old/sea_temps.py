import pandas as pd

def fetch_amo_data():
    # URL of the dataset
    url = 'https://psl.noaa.gov/data/correlation//amon.us.long.data'

    # Read the data, treating -99.99 as NaN
    data = pd.read_csv(url, delim_whitespace=True, skiprows=6, header=None, 
                       names=['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 
                              'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual'],
                       na_values='-99.99')

    # Drop the last 5 rows
    data = data.iloc[:-5]

    # Convert 'Year' column to integers
    data['Year'] = data['Year'].astype(int)

    # Convert 'Jan' to 'Dec' columns to floats
    columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    data[columns] = data[columns].astype(float)

    # Calculate the mean for each row from Jan to Dec and update the Annual column
    data['Annual'] = data.loc[:, 'Jan':'Dec'].mean(axis=1)

    return data

if __name__ == "__main__":
    data = fetch_amo_data()
