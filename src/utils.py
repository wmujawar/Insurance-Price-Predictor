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

        
def generate_html_performance_report(rmse, r2_score, adjusted_r2_score):
    """
    Generate an HTML report containing train performance metrics.
    
    Parameters:
    - title: Title of the report
    - rmse: The RMSE score.
    - r2_score: The R² (R-squared) score.
    - adjusted_r2_score: The Adjusted R² score.
    
    Returns:
    - A string containing the HTML content.
    """
    html_content = f"""
    <p>RMSE: {rmse:.2f}</p>
    <p>R²: {r2_score:.4f}</p>
    <p>Adjusted R²: {adjusted_r2_score:.4f}</p>
    """
    return html_content

