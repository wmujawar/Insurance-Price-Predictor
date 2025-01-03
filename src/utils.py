import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import mlflow


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


def setup_mlflow_experiment(exp_name: str, exp_description: str) -> mlflow.entities.Experiment:
    """
    Set up an MLflow experiment.

    Parameters:
    exp_name (str): The name of the experiment.
    exp_description (str): The description of the experiment.

    Returns:
    mlflow.entities.Experiment: The MLflow experiment object.
    """
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    mlflow.set_tracking_uri(f"file://{parent_dir}/mlruns")
    experiment = mlflow.get_experiment_by_name(exp_name)

    if not experiment:
        mlflow.create_experiment(exp_name, tags={'mlflow.note.content': exp_description})
        experiment = mlflow.get_experiment_by_name(exp_name)
        
    return experiment

