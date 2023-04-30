import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('Assets','train.csv')
    test_data_path:str=os.path.join('Assets',"test.csv")
    clean_data_path:str=os.path.join('Assets',"cleaned_data.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self,cleaned_data_path):
        logging.info("Data Ingestion Initiated")
        try:
            logging.info("Reading CSV Cleaned Data")
            clean_data=pd.read_csv(cleaned_data_path)
            
            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(clean_data,test_size=0.2,random_state=101)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
    
            clean_data.to_csv(self.ingestion_config.clean_data_path)
            if os.path.exists(cleaned_data_path):
                os.remove(cleaned_data_path)

            logging.info("Storing Train and Test Files")
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            logging.info("Data Ingestion Completed")

        except Exception as e:
            raise CustomException(e,sys)