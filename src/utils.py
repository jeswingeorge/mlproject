import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

import dill

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
    
