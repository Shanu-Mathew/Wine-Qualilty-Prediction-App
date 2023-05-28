import os

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import r2_score

from src.utils import evaluate_model,save_object
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('Artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and Test Input Data")
            X_train,Y_train,X_test,Y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                )
            models={
                "KNeighbors" : KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "AdaBoost Classifier":AdaBoostClassifier(),
                
            }
            params={
                "KNeighbors":{
                    'n_neighbors':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                },
                "Decision Tree": {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    #'criterion':['squared_error', 'friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    #'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
            }

            model_report:dict=evaluate_model(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models,param=params)
            #To get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #To get best model name from dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                logging.info('No Best Model Found')
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            best_model.fit(X_train,Y_train)
            predicted=best_model.predict(X_test)
            r2_square=r2_score(Y_test,predicted)
            return r2_square
        
        
        except Exception as e:
            raise CustomException(e,sys)
