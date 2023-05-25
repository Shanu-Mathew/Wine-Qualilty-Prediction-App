import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from imblearn.over_sampling import RandomOverSampler
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
            clean_data_df=pd.read_csv(cleaned_data_path)
            
            target_column_name="quality"
            X_df=clean_data_df.drop(columns=[target_column_name],axis=1)
            Y_df=clean_data_df[target_column_name]
            
            logging.info("Sampling Initiated")
            random_over_sampler=RandomOverSampler(sampling_strategy="not majority")
            sampled_train_set,sampled_test_set=random_over_sampler.fit_resample(X_df,Y_df)

            logging.info("Sampling Done")
            sampled_data=pd.concat([sampled_train_set,sampled_test_set],axis=1)

            logging.info("Train Test Split Initiated")
            train_set,test_set=train_test_split(sampled_data,test_size=0.2,random_state=101)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("Saving the Clean Data")
            clean_data_df.to_csv(self.ingestion_config.clean_data_path,mode='w',index=False)
            if os.path.exists(cleaned_data_path):
                os.remove(cleaned_data_path)

            logging.info("Storing Train and Test Files")
            train_set.to_csv(self.ingestion_config.train_data_path,mode='w',index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,mode='w',index=False, header=True)
            logging.info("Data Ingestion Completed")
            
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)