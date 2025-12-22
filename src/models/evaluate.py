import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate regression model and return metrics.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return y_pred, rmse, r2

