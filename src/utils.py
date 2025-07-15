import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

import dill

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    """
    Function to save an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
        logging.info(f"Object saved at {file_path}")
    
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Function to evaluate multiple models and return a report.
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)  # Fit the model on training data
            logging.info(f"Model {model_name} trained successfully.")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2_score = r2_score(y_train, y_train_pred)
            test_r2_score = r2_score(y_test, y_test_pred)

            r2 = model.score(X_test, y_test)
            mae = np.mean(np.abs(y_test - y_test_pred))
            mse = np.mean((y_test - y_test_pred) ** 2)
            rmse = np.sqrt(mse)

            model_report[model_name] = {
                'train_r2_score': train_r2_score,
                'test_r2_score': test_r2_score,
                'mean_absolute_error': mae,
                'mean_squared_error': mse,
                'root_mean_squared_error': rmse
            }
        
        return model_report
    
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)