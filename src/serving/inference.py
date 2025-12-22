import pandas as pd
import mlflow
import os
import joblib

from src.features.feature_engineering import feature_extraction, add_distance_features
from src.utils.feature_utils import get_feature


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

kmeans_model = joblib.load(
    os.path.join(MODELS_DIR, "Kmeans_model.pkl"))

ridge_model = joblib.load(
    os.path.join(MODELS_DIR, "Ridge_model.pkl"))

TEST_FEATURES = get_feature()


def _transform(df):
    df = df.copy()

    # KMeans clustering
    df['pickup_cluster'] = kmeans_model.predict(df[['pickup_latitude', 'pickup_longitude']].values)
    df['dropoff_cluster'] = kmeans_model.predict(df[['dropoff_latitude', 'dropoff_longitude']].values)


    # Feature engineering
    df = feature_extraction(df)
    df = add_distance_features(df)

    # ترتيب الأعمدة زي التدريب
    df = df[TEST_FEATURES]
    return df


def predict(input_dict):
    
    df = pd.DataFrame([input_dict])
    df = _transform(df)  # مهم جداً
    trip_duration = float(ridge_model.predict(df)[0])
    return trip_duration

