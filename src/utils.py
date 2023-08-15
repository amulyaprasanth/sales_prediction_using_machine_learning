import os
import sys
import pickle
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(params.keys())[i]]

            gs = GridSearchCV(model, param_grid=para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
            train_mae = mean_absolute_error(y_train, train_preds)
            train_r2 = r2_score(y_train, train_preds)

            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_mae = mean_absolute_error(y_test, test_preds)
            test_r2 = r2_score(y_test, test_preds)

            logging.info(f"""   Model: {list(models.keys())[i]}
                                Metrics (train_score, test_score)
                                rmse: {train_rmse, test_rmse}
                                mae : {train_mae, test_mae}
                                r2  : {train_r2, test_r2} """)

            report[list(models.keys())[i]] = test_rmse

        return report

    except Exception as e:
        raise CustomException(e, sys)


