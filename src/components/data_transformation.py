import sys
import os
from dataclasses import dataclass 

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """ 
        This function is responsible for creating a preprocessor object that will be used to transform the data.
        It includes pipelines for numerical and categorical features, handling missing values, scaling, and encoding.
        """
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race/ethnicity', 'parental_level_of_education',
                                   'lunch', 'test_preparation_course']
            ## numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
            )

            logging.info("Numerical columns pipelines created successfully")

            ## categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler())
            ]
            )

            logging.info("Categorical columns pipelines created successfully")

            preprocessor = ColumnTransformer(
                transformers=[ 
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            logging.info("Combined numerical and categorical pipelines and Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function is responsible for initiating the data transformation process.
        It reads the training and testing data, applies the preprocessor, and saves the transformed data.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")

            logging.info("Obtaining preprocessing object")
            # Get the preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score', 'reading_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                ### Function to save the preprocessor object in artifacts folder as a pickle file
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                ### the preprocessing_obj has the preprocessor object with pipelines for numerical and categorical features
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessor object")

            return(train_arr, test_arr, preprocessing_obj)

        except Exception as e:
            raise CustomException(e, sys)
        

