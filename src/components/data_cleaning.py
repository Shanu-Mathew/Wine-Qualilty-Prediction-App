import os
import sys

import pandas as pd
from dataclasses import dataclass     

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataCleaningConfig:
    raw_data=os.path.join('Data','Raw_Data.xlsx')
    cleaned_data_path=os.path.join('Data','Clean_Data.csv')

class DataCleaning:
    def __init__(self):
        self.cleaning_config=DataCleaningConfig()
    def initiate_data_cleaning(self):
        logging.info("Data Cleaning Stage Started")
        try:
            raw_data_df=pd.read_excel(self.cleaning_config.raw_data)
            cleaned_data_df=raw_data_df.dropna()
            logging.info("Nan Rows dropped as only three rows contained")
            cleaned_data_df.to_csv(self.cleaning_config.cleaned_data_path,mode='w',index=False,header=True)
            return self.cleaning_config.cleaned_data_path
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataCleaning()
    cleaned_data_path=obj.initiate_data_cleaning()
    
    data_ingestion=DataIngestion()
    logging.info("Clean Data Path:- {}".format(cleaned_data_path))
    train_data,test_data=data_ingestion.initiate_data_ingestion(cleaned_data_path=cleaned_data_path)
    
    data_transformation=DataTransformation()
    train_arr,test_arr,preprocessor_path1=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))