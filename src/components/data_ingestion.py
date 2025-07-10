import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration Class
    """
    ### Paths for the dataset and train/test splits
    ### raw_data_path,train_data_path and test_data_path are defined as strings
    ### All the outputs of data ingestion will be stored in the 'artifacts' directory
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        ## Initialize the Data Ingestion class
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to initiate data ingestion
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from the CSV file or from databases
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')  ## Line to change if reading from a different source or databases
            logging.info("Dataset read as pandas DataFrame")

            ## Update the column names to replace spaces with underscores
            df.columns = df.columns.str.replace(' ', '_', regex=True)
            logging.info("Column names updated to replace spaces with underscores")

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to CSV")

            # Split the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and test sets created")

            # Save the train and test sets to CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train and test sets saved to CSV files")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path, raw_data_path = obj.initiate_data_ingestion()
    print(f"Train Data Path: {train_data_path}")
    print(f"Test Data Path: {test_data_path}")
    print(f"Raw Data Path: {raw_data_path}")
    
    ### create an object of DataTransformation class
    data_transformation = DataTransformation()
    ### call the initiate_data_transformation method to transform the data
    train_arr, test_arr, preprocessing_obj = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    print(f"Preprocessing Object Path: {DataTransformationConfig().preprocessor_obj_file_path}")


# This code is designed to be run as a script, and it will execute the data ingestion process
# when run directly. It will print the paths of the train, test, and raw data files created during the process.
# The paths are relative to the current working directory and will be created in the 'artifacts' directory.
# Ensure that the 'notebook/data/StudentsPerformance.csv' file exists in the specified path
# before running this script.
# The code handles exceptions using a custom exception class and logs the process using a logging module.
# The data ingestion process includes reading the dataset, saving it to a raw data file,        
# splitting it into train and test sets, and saving those sets to their respective files.
# The code uses the `dataclass` decorator to define the configuration for data ingestion,
# which includes paths for the train, test, and raw data files. 
# The `DataIngestion` class encapsulates the data ingestion logic, and the `initiate_data_ingestion`
# method performs the actual data ingestion steps.
# The code is structured to be modular and reusable, allowing for easy integration into larger data processing pipelines.
# The `if __name__ == "__main__":` block allows the script to be run directly, making it convenient for testing and execution.
# The logging statements provide insights into the progress of the data ingestion process,  
# making it easier to debug and monitor the execution flow.
# The code is designed to be flexible and can be adapted for different datasets and configurations as needed.
# The use of `os.makedirs` ensures that the necessary directories are created if they do not exist,
# preventing errors related to missing directories when saving files.   
# The `train_test_split` function from `sklearn.model_selection` is used to split the dataset into training and testing sets,
# ensuring that the data is randomly divided while maintaining a specified test size.
# The `dataclass` is used to define the configuration for data ingestion, making it easy to manage and extend in the future.
# The code is structured to be clear and maintainable, following best practices for Python programming.
# The use of type hints and default values in the `DataIngestionConfig` class enhances code readability and usability.
# The code is designed to be run in a Jupyter notebook or as a standalone script,
# making it versatile for different development environments.
# The paths for the dataset and output files are defined relative to the current working directory,
# allowing for easy portability and adaptability to different project structures.