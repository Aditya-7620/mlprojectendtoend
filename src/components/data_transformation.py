import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import os

from src.utils import save_object

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor 
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):  
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()  # Fixed: get_dat_transformer_object -> get_data_transformer_object

            target_column_name = "math_score"  # Fixed: name_score -> math_score (assuming standard students performance dataset)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)  # Fixed: train.df -> train_df, drop() syntax
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)  # Fixed: test.df -> test_df
            target_feature_test_df = test_df[target_column_name]  # Fixed: target_feature_train_df -> target_feature_test_df

            logging.info(  # Fixed: logging.Info -> logging.info
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)  # Fixed: preprocessor_obj -> preprocessing_obj
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Fixed: preprocessor_obj -> preprocessing_obj

            train_arr = np.c_[  # Fixed: incomplete np.c_
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[  # Added: test array
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            pickle.dump(preprocessing_obj, open(self.data_transformation_config.preprocessor_obj_file_path, 'wb'))

            save_object(  # Assuming you have utils.save_object method
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)  # Fixed: bare except -> proper exception handling