import pandas as pd
import numpy as np
import os, sys
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping, SensorModel, TransformerResolver
from sensor.utils.main_util import load_object
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.utils.main_util import read_yaml_file

class Prediction:

    def __init__(self, df:pd.DataFrame):

        try:
            self.df = df
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e,sys)
    
    def get_transformer_obj(self):

        try:
            tranfromer_resolver = TransformerResolver()
            if not tranfromer_resolver.is_transformer_exists():
                return "Transformer is not available"
            
            best_transformer_path=tranfromer_resolver.get_latest_transformer_path()
            trans_obj = load_object(file_path=best_transformer_path)
            return trans_obj
        
        except Exception as e:
            raise SensorException(e,sys)
        
    def get_model_obj(self):

        try:
            model_resolver = ModelResolver()
            if not model_resolver.is_model_exists():
                return "Model is not available"
            
            best_model_path = model_resolver.get_best_model_path()
            model_obj = load_object(file_path=best_model_path)
            return model_obj
        
        except Exception as e:
            raise SensorException(e,sys)
    
    def initiate_prediction(self):
        try:
            self.df=self.df.drop(columns=["_id"], axis=1)
            df = self.df.drop(self._schema_config["drop_columns"],axis=1)

            model_obj = self.get_model_obj()
            trans_obj = self.get_transformer_obj()

            model = SensorModel(preprocessor=trans_obj, model=model_obj)
            y_pred = model.predict(df)
            df['predicted_column'] = y_pred
            df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
            
            return df
        
        except Exception as e:
            raise SensorException(e,sys)