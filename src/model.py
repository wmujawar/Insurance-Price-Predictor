import os
from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickle

from utils import r2_and_adjusted_r2_score as score, root_mean_squared_error as rmse

import json
import xgboost as xgb

class Model:
    """
    A class used to represent a Model for insurance premium prediction.
    
    Attributes
    ----------
    folder : str
        The folder where the data file is located.
    file : str
        The name of the data file.
        
    Methods
    -------
    load_data() -> pd.DataFrame:
        Loads the data from the specified folder and file.
    split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        Splits the data into training and testing sets based on age categories.
    create_model(alpha: float, l1_ratio: float) -> Pipeline:
        Creates a machine learning model pipeline with specified hyperparameters.
    evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
        Evaluates the model performance using RMSE, R2, and adjusted R2 scores.
    save_metrics(metrics: dict) -> None:
        Saves the evaluation metrics to a JSON file.
    save_model(model: Pipeline, model_name: str) -> None:
        Saves the trained model to a file.
    """
    
    def __init__(self):
        self.parent_dir = os.path.abspath(os.path.join(os.getcwd()))
        
        
    def load_data(self, folder: str, file: str) -> pd.DataFrame:
        """
        Loads the data from the specified folder and file.
        
        Parameters
        ----------
        folder : str
            The folder where the data file is located in the 'data' folder of the project.
        file : str
            The name of the data file.
        
        Returns
        -------
        pd.DataFrame
            The loaded data as a pandas DataFrame.
        """
        
        raw_data_dir = os.path.join(self.parent_dir, 'data', folder)
        return pd.read_csv(os.path.join(raw_data_dir, file))
        
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data into training and testing sets based on age categories.
        
        Parameters
        ----------
        df : pd.DataFrame
            The data to be split into training and testing sets.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            The training and testing features and target variables.
        """

        # Define the age categories
        age_bins = [18, 25, 40, 55, np.inf]
        age_labels = ['Young Adult', 'Adult', 
                    'Middle Aged Adults', 'Senior']

        age_category = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

        # Split the data into train and test sets
        return train_test_split(df.drop(columns=['PremiumPrice']), df['PremiumPrice'],
                                        stratify=age_category, test_size=0.2, random_state=42)
        
        
        
    def create_model(self, alpha: float, l1_ratio: float) -> Pipeline:
        """
        Creates a machine learning model pipeline with specified hyperparameters.
        
        Parameters
        ----------
        alpha : float
            The alpha parameter for the ElasticNet final estimator.
        l1_ratio : float
            The l1_ratio parameter for the ElasticNet final estimator.
            
        Returns
        -------
        Pipeline
            The machine learning model pipeline.
        """
        
        # Define the base learners and the final estimator
        base_learner = [
            ('gb1', GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.2, loss='huber')),
            ('gb2', GradientBoostingRegressor(n_estimators=80, max_depth=5, learning_rate=0.3, loss='huber')),
            ('gb3', GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.2, loss='huber')),
            ('rd1', RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1)),
            ('rd2', RandomForestRegressor(n_estimators=110, max_depth=6, n_jobs=-1)),
            ('xg', xgb.XGBRegressor(max_depth=4, n_estimators=80, learning_rate=0.05, reg_alpha=0.7, reg_lambda=0.1))
        ]

        final_estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)

        # Define the preprocessor
        numeric_columns = ['Age', 'Height', 'Weight']
        binary_columns = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']

        preprocessor = ColumnTransformer([
            ('scaler', StandardScaler(), numeric_columns),
            ('passthrough', 'passthrough', binary_columns)
        ])

        # Define the pipeline
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', StackingRegressor(estimators=base_learner, final_estimator=final_estimator))
        ])
        
        return pipeline
    
    
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, n: int, p: int) -> Tuple[float, float, float]:
        """
        Evaluates the model performance using RMSE, R2, and adjusted R2 scores.
        
        Parameters
        ----------
        y_true : pd.Series
            The true target values.
        y_pred : pd.Series
            The predicted target values.
        n : int
            The number of samples.
        p : int
            The number of features.
            
        Returns
        -------
        Tuple[float, float, float]
            The RMSE, R2, and adjusted R2 scores.
        """
        rmse_score = rmse(y_true, y_pred)
        r2_score, adj_r2_score = score(y_true, y_pred, n, p)
                
        return rmse_score, r2_score, adj_r2_score
    
    
    def save_metrics(self, metrics: dict, file: str) -> None:
        """
        Saves the evaluation metrics to a JSON file.
        
        Parameters
        ----------
        metrics : dict
            The evaluation metrics to be saved.
        file : str
            The name of the JSON file to save the metrics. Do not provide file extension.
        """
        
        metrics_dir = os.path.join(self.parent_dir, 'output')
        metrics_path = os.path.join(metrics_dir, f'{file}.json')
        
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
    
    
    def save_model(self, model: Pipeline, model_name: str) -> None:
        """
        Saves the trained model to a file.
        
        Parameters
        ----------
        model : Pipeline
            The trained machine learning model pipeline.
        model_name : str
            The name of the file to save the model.
        """
        
        model_dir = os.path.join(self.parent_dir, 'output')
        model_path = os.path.join(model_dir, model_name)
        
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)



if __name__ == '__main__':
    
    model = Model()
    
    # Load the data
    data = model.load_data('raw', 'insurance.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = model.split_data(data)
    
    # Create the model
    model_pipeline = model.create_model(0.7, 0.3)
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model_pipeline.predict(X_train)
    y_test_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    train_rmse, train_r2, train_adj_r2 = model.evaluate(y_train, y_train_pred, X_train.shape[0], X_train.shape[1])
    test_rmse, test_r2, test_adj_r2 = model.evaluate(y_test, y_test_pred, X_test.shape[0], X_test.shape[1])
    
    # Save the evaluation metrics
    train_metrics = {
        'RMSE': train_rmse,
        'R2': train_r2,
        'Adjusted R2': train_adj_r2
    }
    
    test_metrics = {
        'RMSE': test_rmse,
        'R2': test_r2,
        'Adjusted R2': test_adj_r2
    }
    
    metrix = {
        'Train Metrics': train_metrics,
        'Test Metrics': test_metrics
    }
    
    model.save_metrics(metrix, 'metrics')
    
    # Save the model
    model.save_model(model_pipeline, 'model.pkl')