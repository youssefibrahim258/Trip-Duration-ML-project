from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(scaler):
    """
    Build a ColumnTransformer for preprocessing features.

    Args:
        scaler: A scikit-learn scaler instance (e.g., StandardScaler(), MinMaxScaler()).

    Returns:
        ColumnTransformer that applies one-hot encoding to categorical features
        and scaling to numeric features.
    """
    numeric_features = ['distance_haversine', 'distance_dummy_manhattan', 'bearing']
    categorical_features = [
        'passenger_count', 'store_and_fwd_flag', 'vendor_id', 'pickup_cluster', 'dropoff_cluster',
        'DayofMonth', 'dayofweek', 'month', 'hour', 'dayofyear'
    ]
    
    column_transformer = ColumnTransformer(
        [
            ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ('scaling', scaler, numeric_features)
        ],
        remainder='passthrough'
    )
    
    return column_transformer

