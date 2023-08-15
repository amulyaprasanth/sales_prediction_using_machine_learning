import os
import sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, load_object

from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_preprocessor_obj(self):
        '''
        This function is responsible for data tranformation
        '''
        try:
            num_features =  ["ID", "YEAR", "CRITICS_POINTS", "USER_POINTS"]
            cat_features = ["CONSOLE", "CATEGORY", "PUBLISHER", "RATING"]
            
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oneHotEncoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
                
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test dataset completed")
            logging.info("Getting preprocessor object")
            
            preprocessor_obj = self.get_preprocessor_obj()
            
            target_column = "SalesInMillions"
            
            input_train_features_df = train_df.drop(target_column, axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_test_features_df = test_df.drop(target_column, axis=1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object on training and test dataframe")
            
            input_features_train_arr = preprocessor_obj.fit_transform(input_train_features_df)
            input_feature_test_arr = preprocessor_obj.transform(input_test_features_df)
                        
            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df) ]
            
            logging.info("Saving preprocessor object")
            
            save_object(self.data_tranformation_config.preprocessor_obj_path,
                        preprocessor_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_path
            )
        
        except Exception as e:
          raise CustomException(e, sys)
        
        