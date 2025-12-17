import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utiles import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("spliting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "logistic Regrestion" : LogisticRegression(max_iter=1000)
            }

            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test,y_test=y_test, models = models )


            best_model_score = max(model_report.values())


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)


            Final_accuracy = accuracy_score(y_test,predicted)
            return Final_accuracy 
                                                        
                                                        
        except Exception as e:
            raise CustomException (e,sys)
        


