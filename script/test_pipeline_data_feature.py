import pandas as pd 
from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import build_feature


TRAIN_PATH=r"Y:\01_ML\Projects\00_End_to_End\01_Trip_Duration\Data\Split-Sample\train.csv"
VAL_PATH=r"Y:\01_ML\Projects\00_End_to_End\01_Trip_Duration\Data\Split-Sample\val.csv"

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

















