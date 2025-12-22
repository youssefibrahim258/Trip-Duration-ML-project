import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_pred(y_true, y_pred):
    """
    Returns Actual vs Predicted figure.
    """
    fig = plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        linestyle="--"
    )
    plt.xlabel("Actual Trip Duration")
    plt.ylabel("Predicted Trip Duration")
    plt.title("Actual vs Predicted ")

    return fig


def plot_residuals(y_true, y_pred):
    """
    Returns Residuals plot figure.
    """
    residuals = y_true - y_pred

    fig = plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Trip Duration")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")

    return fig



