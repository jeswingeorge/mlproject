import os
import sys

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    """
    Model Trainer Configuration Class
    """
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    # model_report_file_path: str = os.path.join('artifacts', 'model_report.txt')

class ModelTrainer:
    def __init__(self):
        """
        Initialize the Model Trainer class with paths for saving the trained model and model report.
        """
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Method to initiate model training
        """
        try:
            logging.info("Split train and test data into features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            ### Dictionary of models to train
            logging.info("Initiating model training with various regression models")
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor()
            }

            params={
                "LinearRegression":{},

                "Lasso": {
                    'alpha': [0.1, 0.01, 0.001, 0.0001],
                    'max_iter': [1000, 2000, 3000]
                },

                "Ridge": {
                    'alpha': [0.1, 0.01, 0.001, 0.0001],
                    'max_iter': [1000, 2000, 3000]
                },

                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },

                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                               models=models, params=params)
            # print(model_report)
            logging.info("Model evaluation completed")
            
            # Find the best model based on R2 score, RMSE or any other metric
            logging.info("Finding the best model based on RMSE")
            ## min() finds the key (model name) in model_report with the smallest value of 'root_mean_squared_error'
            best_model_name = min(model_report, key=lambda x: model_report[x]['root_mean_squared_error'])
            best_model_score = model_report[best_model_name]['root_mean_squared_error']
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with RMSE: {best_model_score}")
            print(f"Best Model: {best_model_name} with RMSE: {best_model_score}")
            print(f"train_r2_score for {best_model_name}: {model_report[best_model_name]['train_r2_score']}")
            print(f"test_r2_score for {best_model_name}: {model_report[best_model_name]['test_r2_score']}")

            logging.info("Saving the best model to artifacts directory")   
            # Save the best model
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)

