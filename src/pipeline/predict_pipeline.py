import os
import sys
import numpy as np
import pandas as pd

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.metrics import confusion_matrix as cm

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
            logging.info('Preprocessor Loading for Predict Pipeline')
            preprocessor=load_object(self.predict_model.best_preprocessor_path)
            data_scaled=preprocessor.transform(data)

            #Preprocessed Data Prediction
            logging.info('Model Loading for Predict Pipeline')
            model=load_object(self.predict_model.best_model_path)
            logging.info('Input Value Prediction')
            model_pred=model.predict(data_scaled)
            model_prob=model.predict_proba(data_scaled)
            
            model_prob=pd.DataFrame(model_prob.reshape(1,-1), columns=np.array([3,4,5,6,7,8]))

            return model_pred,model_prob
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    data=pd.read_csv('Assets/test.csv')
    data_drp=data.drop('quality',axis=1)
    logging.info('Predict Pipeline Working')
    predicted=PredictPipeline.initiate_prediction(PredictPipeline(),data_drp)
    logging.info('Successful Prediction')
    actual=data['quality']
    print(cm(actual,predicted))