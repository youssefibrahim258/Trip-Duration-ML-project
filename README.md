# Trip Duration Predictor

## Project Overview
The **Trip Duration Predictor** aims to estimate the duration of taxi rides by analyzing various features, including geographic coordinates, time of day, and other relevant factors. Utilizing machine learning techniques, particularly **Ridge regression**, this project investigates historical trip data to identify patterns, enabling accurate predictions for future scenarios. Recent enhancements include:

- Integration with **MLflow** for tracking parameters, metrics, and plots.
- Deployment using **FastAPI** and **Docker** for scalable serving.
- Utilization of **argparse** for command-line parameter parsing.

## Contents
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Cluster Feature Extraction](#cluster-feature-extraction)
- [Data Visualization](#data-visualization)
- [Data Transformation](#data-transformation)
- [Machine Learning Model](#machine-learning-model)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Deployment](#deployment)
- [Model Evaluation and Visualization](#model-evaluation-and-visualization)

## Data Cleaning
The initial step focused on improving data quality by addressing outliers. This process involved removing or managing outliers, enhancing the model’s accuracy, and ensuring data consistency.

## Feature Engineering
In addition to geographical data (longitude and latitude), new features such as Haversine distance, Manhattan distance, and directional indicators were created, significantly boosting the model's predictive capabilities.

## Cluster Feature Extraction
The **MiniBatchKMeans** algorithm was employed to cluster pickup and dropoff coordinates into 'n' groups. This clustering facilitated the addition of new features, including **pickup_cluster** and **dropoff_cluster**, enriching the dataset.

## Data Visualization
Data visualization techniques were employed to explore categorical data and identify trends. For numeric data, a right-skewed distribution was observed, prompting a log transformation to enhance data analysis and understanding.

## Data Transformation
Using **ColumnTransformer**, **OneHotEncoder** was applied to categorical features, while **StandardScaler** was used for numeric features. Additionally, new time-based features were derived from datetime columns, such as extracting the **day** and **hour** from timestamps.

## Machine Learning Model
A machine learning model was developed utilizing a **Pipeline** that integrates:
- **ColumnTransformer** for preprocessing
- **PolynomialFeatures** (degree 2) to account for feature interactions
- **Ridge regression** with an alpha value of 50, optimizing model performance

This pipeline improves the model's generalization capability, leading to enhanced predictions and stability.

## Experiment Tracking with MLflow
The project employs **MLflow** for tracking parameters, metrics, and plots. Key features include:
- Logging hyperparameters such as `alpha`, `scaler`, and `poly_degree`.
- Capturing model performance metrics including RMSE and R² for both training and validation datasets.
- Saving visualizations of actual vs. predicted values and residual plots.

Here is a screenshot demonstrating the use of MLflow for tracking experiments:

![MLflow UI Screenshot](ss.png)

## Deployment
The model is deployed using **FastAPI** and **Docker**, allowing for scalable and efficient serving of predictions. This setup facilitates easy integration into web applications, providing a robust API for users.

## Model Evaluation and Visualization

### Actual vs Predicted Plot
The **Actual vs Predicted** plot visualizes the relationship between the true trip durations and the predicted values. 

![Actual vs Predicted](path/to/your/mlflow/plots/train/train_actual_vs_pred.png)

### Residuals Plot
The **Residuals Plot** illustrates the residuals (the difference between actual and predicted values) against the predicted trip durations. 

![Residuals Plot](path/to/your/mlflow/plots/train/residuals_train.png)



## Results
| Metric            | Training R²          | Training RMSE         | Validation R²        | Validation RMSE      | Test R²             | Test RMSE           |
|-------------------|----------------------|-----------------------|-----------------------|----------------------|---------------------|---------------------|
| **Values**        | 0.7871               | 0.3531                | 0.7312                | 0.4148               | 0.7353           
