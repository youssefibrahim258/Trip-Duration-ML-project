import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures


from src.data.load_data import load_data
from src.data.preprocess import clean_data
from src.features.feature_engineering import build_feature
from src.features.preprocessing import build_preprocessor
from src.utils.feature_utils import get_feature
from src.utils.model_utils import get_model
from src.models.train import train_pipeline
from src.models.evaluate import evaluate_model
from src.visualization.plots import plot_actual_vs_pred,plot_residuals


def main(args):

    # Step 1 : Load the Data
    df_train=load_data(args.train_path)
    df_val=load_data(args.val_path)

    print("Done step1")
    # Step 2 : Clean the train data
    df_train=clean_data(df_train)
    print("Done step2")

    # Step 3 : Build a new features
    df_train,df_val,kmeans_model=build_feature(df_train,df_val)
    print("Done step3")

    # Step 4 : Preprocessing
    if args.scaler =="standardScaler":
        scaler=StandardScaler()
    else:
        scaler=MinMaxScaler()

    column_transformer=build_preprocessor(scaler)
    print("Done step4")

    # Step 5 : Get the selected train features and the selected model
    feature_selected=get_feature()
    selected_model=get_model(args.model)
    target= "log_trip_duration"

    # Step 6 : Prebare the pipeline
    pipeline=Pipeline(steps=[
        ("ohe",column_transformer),
        ("poly",PolynomialFeatures(args.poly_degree)),
        ("regression",selected_model(alpha=args.alpha))
    ])
    print("Done step6")

    # Step 7 : Train the model and evaluate + Mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")   # Use SQLite DB instead of mlruns

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="Trip_Duration") as run:
        mlflow.set_tag("model", "Ridge")
        print("start_train")
        model=train_pipeline(pipeline,df_train,feature_selected,target)
        mlflow.log_params({
            "alpha": args.alpha,
            "scaler": args.scaler,
            "poly_degree": args.poly_degree})
        
        print("end_of_train")

        # Log the train Metrix and Figure
        x_train = df_train[feature_selected]
        y_train = df_train[target]
        y_train_pred, rmse_train, r2_train=evaluate_model(model,x_train,y_train)
        mlflow.log_metrics({
            "rmse_train":rmse_train,
            "r2_train":r2_train})

        plot_actual_vs_pred_train=plot_actual_vs_pred(y_train,y_train_pred)
        mlflow.log_figure(plot_actual_vs_pred_train, "plots/train/train_actual_vs_pred.png")

        plot_residuals_train=plot_residuals(y_train,y_train_pred)
        mlflow.log_figure(plot_residuals_train,"plots/train/residuals_train.png")

        # Log the val Metrix and Figure
        x_val = df_val[feature_selected]
        y_val = df_val[target]
        y_val_pred, rmse_val, r2_val=evaluate_model(model,x_val,y_val)
        mlflow.log_metrics({
            "rmse_val":rmse_val,
            "r2_val":r2_val})


        mlflow.sklearn.log_model(model, "Ridge_Model", registered_model_name="Ridge_Model")

        mlflow.sklearn.log_model(kmeans_model, "Kmeans_Model", registered_model_name="Kmeans_Model")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Trip Duration Project + mlflow")
    parser.add_argument('--train_path',type=str,default=r"Y:\01_ML\Projects\00_End_to_End\01_Trip_Duration\Data\train.csv",help='train_path')
    parser.add_argument('--val_path',type=str,default=r"Y:\01_ML\Projects\00_End_to_End\01_Trip_Duration\Data\val.csv",help='val_path')

    parser.add_argument('--scaler',type=str,default="standardScaler",
                        choices=["standardScaler","MinMaxScaler"],
                        help='Choose scaler: "standardScaler","MinMaxScaler"')
    
    parser.add_argument('--poly_degree',type=int,default=2,help='Degree of Polynomial feature')
    parser.add_argument('--model',type=int,default=1,help='1 for Ridge')
    parser.add_argument("--alpha",type=int,default=50)
    parser.add_argument("--experiment", type=str, default="Trip_Duration2")

    args=parser.parse_args()

    main(args)
