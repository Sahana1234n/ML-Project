import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object , evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_trainer(self,train_array,test_array):    
        try:
            logging.info("Splitting training and testing input data")
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]  
                  )
            
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose=False),
                "AdaBoost Regressor" : AdaBoostRegressor(),
            }
            param_grid = {
                "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
            },
            "Decision Tree": {
            "criterion": ["squared_error", "friedman_mse"],
            "max_depth": [None, 10],
            },
            "Gradient Boosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
            },
            "Linear Regression": {},
            "K-Neighbors Classifier": {
            "n_neighbors": [3, 5, 7],
            },
            "XGBClassifier": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
        },
            "CatBoosting Regressor": {
            "depth": [6, 8],
            "learning_rate": [0.05, 0.1],
        },
            "AdaBoost Classifier": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
        },
        }


            model_report:dict=evaluate_models(X_train =X_train ,y_train = y_train , X_test = X_test , y_test = y_test ,models=models,param_grid=param_grid)

            ## To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict 

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and tetsing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            final_r2_score = r2_score(y_test , predicted)
            return final_r2_score
        

        except Exception as e:
            raise CustomException (e,sys) 

