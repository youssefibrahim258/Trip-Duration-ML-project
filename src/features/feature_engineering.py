import numpy as np 
import pandas as pd 
from sklearn.cluster import MiniBatchKMeans


def _cluster_features(train, n=10, random_state=42):
    """
    Create pickup and dropoff cluster features using MiniBatch KMeans.

    Args:
        train: Input DataFrame .
        n: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Updated DataFrame with cluster features and the fitted KMeans model.
    """

    coords = np.vstack((
        train[['pickup_latitude', 'pickup_longitude']].values,
        train[['dropoff_latitude', 'dropoff_longitude']].values
    ))

    np.random.seed(random_state)
    sample_ind = np.random.permutation(len(coords))[:500000]

    kmeans = MiniBatchKMeans(
        n_clusters=n,
        batch_size=10000,
        random_state=random_state
    ).fit(coords[sample_ind])

    train['pickup_cluster'] = kmeans.predict(
        train[['pickup_latitude', 'pickup_longitude']].values
    )
    train['dropoff_cluster'] = kmeans.predict(
        train[['dropoff_latitude', 'dropoff_longitude']].values
    )

    return train, kmeans

def feature_extraction(df):
    """
    Extract time-based features from the pickup datetime.

    Args:
        train: Input DataFrame.

    Returns:
        DataFrame with extracted temporal features.
    """
    df.drop(columns=['id'], inplace=True)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['DayofMonth'] = df['pickup_datetime'].dt.day
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear

    return df

def _haversine_array(lat1, lng1, lat2, lng2): 
    """
    Calculate the distance (in km) between two geographic points.

    Args:
        lat1: Latitude of the first point.
        lng1: Longitude of the first point.
        lat2: Latitude of the second point.
        lng2: Longitude of the second point.

    Returns:
        Distance between the two points in kilometers.
    """

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 
    AVG_EARTH_RADIUS = 6371  # in km 
    lat = lat2 - lat1 
    lng = lng2 - lng1 
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 
    return h

def _bearing_array(lat1, lng1, lat2, lng2):
    """
    Calculate the bearing angle from the first point to the second point.

    Args:
        lat1: Latitude of the starting point.
        lng1: Longitude of the starting point.
        lat2: Latitude of the destination point.
        lng2: Longitude of the destination point.

    Returns:
        Bearing angle in degrees.
    """
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))

def _dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    """
    Approximate Manhattan distance using Haversine calculations.

    Args:
        lat1: Latitude of the starting point.
        lng1: Longitude of the starting point.
        lat2: Latitude of the destination point.
        lng2: Longitude of the destination point.

    Returns:
        Approximate Manhattan distance in kilometers.
    """
    a = _haversine_array(lat1, lng1, lat1, lng2)
    b = _haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def add_distance_features(df):
    """
    Add distance and bearing features to the DataFrame.

    Args:
        df: Input DataFrame with pickup and dropoff coordinates.

    Returns:
        DataFrame with distance_haversine, bearing, and distance_dummy_manhattan features.
    """

    df['distance_haversine'] = _haversine_array(df['pickup_latitude'].values,
                                                  df['pickup_longitude'].values,
                                                  df['dropoff_latitude'].values,
                                                  df['dropoff_longitude'].values)


    df['bearing'] = _bearing_array(df['pickup_latitude'].values,
                                     df['pickup_longitude'].values,
                                     df['dropoff_latitude'].values,
                                     df['dropoff_longitude'].values)

   
    df['distance_dummy_manhattan'] = _dummy_manhattan_distance(df['pickup_latitude'].values,
                                                                 df['pickup_longitude'].values,
                                                                 df['dropoff_latitude'].values,
                                                                 df['dropoff_longitude'].values)
    
    return df



def log_transform(df):
    """
    Apply log transformation to distance features and trip duration, 
    and drop original columns no longer needed.

    Args:
        df: Input DataFrame with distance features and trip_duration.

    Returns:
        DataFrame with log-transformed features.
    """
    df['distance_haversine'] = np.log1p(df['distance_haversine'])
    df['distance_dummy_manhattan'] = np.log1p(df['distance_dummy_manhattan'])
    df['log_trip_duration'] = np.log1p(df['trip_duration'])
    df.drop(columns=['trip_duration', 'pickup_datetime'], inplace=True)
    return df


def build_feature(train_data,val_data):
    """
    Build features for training and validation datasets.

    Steps:
    1. Cluster pickup and dropoff locations using KMeans.
    2. Extract time-based features from pickup_datetime.
    3. Compute distances and bearing features.
    4. Apply log transformation and drop unused columns.

    Args:
        train_data: Training DataFrame with raw features.
        val_data: Validation DataFrame with raw features.

    Returns:
        Tuple of transformed (train_data, val_data) DataFrames ready for modeling.
    """

    # Step 1  
    train_data, kmeans_model = _cluster_features(train_data, n=100, random_state=42)
    val_data['pickup_cluster'] = kmeans_model.predict(val_data[['pickup_latitude', 'pickup_longitude']].values)
    val_data['dropoff_cluster'] = kmeans_model.predict(val_data[['dropoff_latitude', 'dropoff_longitude']].values)
    

    # Step 2 
    train_data=feature_extraction(train_data)
    val_data=feature_extraction(val_data)


    # Step 3
    train_data = add_distance_features(train_data)
    val_data = add_distance_features(val_data)


    # Step 4
    train_data = log_transform(train_data)
    val_data = log_transform(val_data) 


    return train_data,val_data,kmeans_model



    