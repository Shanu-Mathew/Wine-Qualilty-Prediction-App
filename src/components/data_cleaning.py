import os
import sys

import pandas as pd
from dataclasses import dataclass     

from src.logger import logging
from src.exception import CustomException
@dataclass
class DataCleaningConfig:
    raw_data="D:\Documents\Projects\Wine-Qualilty-Analysis\Data\Raw_Data.xlsx"
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
            cleaned_data_df.to_csv(self.cleaning_config.cleaned_data_path,index=False,header=True)
            return self.cleaning_config.cleaned_data_path
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataCleaning()
    cleaned_data=obj.initiate_data_cleaning()
    print(cleaned_data)

