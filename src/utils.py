import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def r2_and_adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n: int, p: int) -> tuple:
    """
    Calculate the R² score and adjusted R² score.

    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.
    n (int): Number of observations.
    p (int): Number of predictors.

    Returns:
    tuple: R² score and Adjusted R² score.
    """
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return r2, adj_r2


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the root mean squared error (RMSE).

    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Predicted values.

    Returns:
    float: RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
