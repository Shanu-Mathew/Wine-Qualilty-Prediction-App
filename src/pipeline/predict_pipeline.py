import os
import sys

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass


@dataclass
class PredictPipelineConfig:
    best_model_path=os.path.join('Artifacts','model.pkl')
    best_preprocessor_path=os.path.join('Artifacts','preprocessor.pkl')
class PredictPipeline:
    def __init__(self):
        self.predict_model=PredictPipelineConfig()
    def initiate_prediction(self,data):
        try:
            #Input Data Preprocessing
            preprocessor=load_object(self.predict_model.best_preprocessor_path)
            data_scaled=preprocessor.transform(data)

            #Preprocessed Data Prediction
            model=load_object(self.predict_model.best_model_path)
            model_pred=model.predict(data_scaled)
            model_prob=model.predict_proba(data_scaled)
            return model_pred,model_prob
        except Exception as e:
            raise CustomException(e,sys)