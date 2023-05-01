
# Wine Quality Prediction

This project explores the relationship between wine attributes and quality ratings using Exploratory Data Analysis (EDA), GridSearch, Cross Validation, and Random Forest algorithms. The goal is to build a model that can predict the quality of a wine based on its physical and chemical characteristics.

## Data
The data used in this project comes the dataset taken from Kaggle. The dataset contains 1599 samples of red wine and their corresponding quality ratings, ranging from 0 (very bad) to 10 (very excellent). There are 11 attributes in total, including fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## Analysis
The analysis starts with EDA to understand the distribution of each attribute and its relationship with the quality rating. The distributions are visualized using histograms, boxplots, and scatterplots. The correlations between attributes are also explored using correlation matrices and heatmaps.

After EDA, GridSearch is used to find the optimal hyperparameters for the Random Forest algorithm. Cross Validation is used to validate the performance of the model and avoid overfitting. The best model is chosen based on its accuracy and r2_score, which measures how well the model fits the data.

## Results
The Random Forest algorithm performs well on the wine quality dataset, with a score of 0.71 and r2_score of 0.38. The most important attributes for predicting wine quality are alcohol, volatile acidity, and sulphates, which have the highest feature importances in the model. The analysis also shows that there is a negative correlation between volatile acidity and wine quality, while there is a positive correlation between alcohol and wine quality.

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>

## Libraries
The following Python libraries were used in this project:

pandas\
numpy\
matplotlib\
seaborn\
scikit-learn

## Conclusion
This project demonstrates the use of EDA, GridSearch, and Random Forest algorithms in predicting wine quality based on physical and chemical characteristics. The Random Forest model performs well on the wine quality dataset and can be used to predict the quality of a wine with reasonable accuracy. Further analysis could include exploring other machine learning algorithms or analyzing the wine quality dataset separately for red and white wines.
