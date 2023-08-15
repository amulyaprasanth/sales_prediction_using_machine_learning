import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = ModelTrainerConfig().trained_model_file_path
            preprocessor_path = DataTransformationConfig.preprocessor_obj_path
            print("Before loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 id: int,
                 year: int,
                 console:str,
                 category: str,
                 publisher: str,
                 rating: str,
                 critics_points: float,
                 user_points: float):
        self.id = id
        self.year = year
        self.console = console
        self.category = category
        self.publisher = publisher
        self.rating = rating
        self.critics_points = critics_points
        self.user_points = user_points

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "ID": [self.id],
                "YEAR": [self.year],
                "CONSOLE": [self.console],
                "CATEGORY": [self.category],
                "PUBLISHER": [self.publisher],
                "RATING": [self.rating],
                "CRITICS_POINTS": [self.critics_points],
                "USER_POINTS": [self.user_points]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
