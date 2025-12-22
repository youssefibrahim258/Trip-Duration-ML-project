
def get_feature():
    """
    Return the list of feature names for the dataset.

    Returns:
        List of all feature column names (categorical + numeric).
    """
    numeric_features = ['distance_haversine', 'distance_dummy_manhattan', 'bearing']
    categorical_features = ['passenger_count', 'store_and_fwd_flag', 'vendor_id', 'pickup_cluster', 'dropoff_cluster',
                            'DayofMonth', 'dayofweek', 'month', 'hour', 'dayofyear']
    train_features = categorical_features + numeric_features

    return train_features

