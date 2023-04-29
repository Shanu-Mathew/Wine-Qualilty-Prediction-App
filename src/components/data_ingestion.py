import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('Assets','train.csv')
    test_daata_path:str=os.path.join('Assets',"test.csv")
    raw_data_path:str=os.path.join('Assets',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            