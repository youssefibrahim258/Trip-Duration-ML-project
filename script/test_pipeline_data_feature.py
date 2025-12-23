"""
Test script for data loading, preprocessing, and feature engineering pipeline.
Used to validate train/validation data transformations.
"""

import pandas as pd 
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import build_feature


TRAIN_PATH=r"Data_Sets\train.csv"
VAL_PATH=r"Data_Sets\test.csv"

def main():
    # 1:Load Data
    df_train=load_data(TRAIN_PATH)
    df_val=load_data(VAL_PATH)

    print(f"Train Data loaded. Shape : {df_train.shape}")
    print(f"Val Data loaded. Shape : {df_val.shape}")

    # 2:preprocess 
    df_train=clean_data(df_train)
    print(f"Train Data after preprocessing. Shape : {df_train.shape}")
    
    # 3:build features
    df_train,df_val,_=build_feature(df_train,df_val)
    print(f"Train data after build features shape : {df_train.shape}")
    print(f"Val data after build features shape : {df_val.shape}")

    print(df_train.columns)

if __name__ == "__main__":
    main()

