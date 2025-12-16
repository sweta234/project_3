import sys
from  dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose  import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utiles import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_features = [
                'Sex',
                'ChestPainType',
                'RestingECG',
                'ExerciseAngina',
                'ST_Slope'
            ]

            categorical_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info("Categorical columns encoding started")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_path = pd.read_csv(test_path)

            logging.info("Read train and test data  completed")

            logging.info("Obtaining preprocessing object   ")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_columns = 'HeartDisease'
            categorical_columns = ['Sex',
                'ChestPainType',
                'RestingECG',
                'ExerciseAngina',
                'ST_Slope']
            
            input_feature_train_df = train_df.drop(columns=[target_columns] , axis = 1)
            target_feature_train_df = train_df[target_columns]
            
            input_feature_test_df = train_df.drop(columns=[target_columns] , axis = 1)
            target_feature_test_df = train_df[target_columns]


            logging.info(

                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df  = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("saved preprocessing object")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )



        except Exception as e:
            raise CustomException (e,sys)