
import numpy as np 
import pandas as pd 

def clean_data(df)->pd.DataFrame:
    """
    Clean the training dataset by:
    - Removing outliers in 'trip_duration' 
    - Filtering invalid coordinates for pickup and dropoff
    - Replacing rows with passenger_count == 0 with NaN and dropping them

    Args: 
        train (pd.DataFrame): Input training dataset

    Returns: 
        pd.DataFrame: Cleaned dataset
    """

    m = np.mean(df['trip_duration'])
    s = np.std(df['trip_duration'])
    df = df[df['trip_duration'] <= m + 2 * s]
    df = df[df['trip_duration'] >= m - 2 * s]

    df = df[df['pickup_longitude'] <= -73.75]
    df = df[df['pickup_longitude'] >= -74.03]
    df = df[df['pickup_latitude'] <= 40.85]
    df = df[df['pickup_latitude'] >= 40.63]
    df = df[df['dropoff_longitude'] <= -73.75]
    df = df[df['dropoff_longitude'] >= -74.03]
    df = df[df['dropoff_latitude'] <= 40.85]
    df = df[df['dropoff_latitude'] >= 40.63]

    df[df['passenger_count'] == 0] = np.nan
    df.dropna(axis=0, inplace=True)
    return df




