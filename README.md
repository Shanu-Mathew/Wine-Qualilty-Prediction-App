
# Wine Quality Prediction

This project explores the relationship between wine attributes and quality ratings using Exploratory Data Analysis (EDA), GridSearch, Cross Validation, and Random Forest algorithms. The goal is to build a model that can predict the quality of a wine based on its physical and chemical characteristics.

## Data
The data used in this project comes the dataset taken from Kaggle. The dataset contains 1599 samples of red wine and their corresponding quality ratings, ranging from 0 (very bad) to 10 (very excellent). There are 11 attributes in total, including fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

## Analysis
To find the best model for predicting wine quality, several machine learning algorithms were tested on the dataset, including Decision Tree, Random Forest, K-Nearest Neighbors, and Boosting Methods. For each algorithm, GridSearch was used to find the optimal hyperparameters.

After testing and comparing the performance of each model, Random Forest was found to be the best algorithm for the wine quality prediction task. GridSearch was used to find the optimal hyperparameters for the Random Forest algorithm. 

Cross Validation was used to validate the performance of the Random Forest model and avoid overfitting. The best model was chosen based on its r2_score, which measures how well the model fits the data.

Additionally, oversampling was performed using the imblearn library, which improved the accuracy of the model. The final model achieved an accuracy of 0.95 and an r2 score of 0.96 on a random train-test split.

The analysis also shows that the most important attributes for predicting wine quality are alcohol, volatile acidity, and sulphates, which have the highest feature importances in the model. The analysis also shows that there is a negative correlation between volatile acidity and wine quality, while there is a positive correlation between alcohol and wine quality.

## Oversampling
To address the class imbalance problem in the dataset, we used the imblearn library to oversample the minority class (quality ratings of 3, 4, and 5). The oversampling technique used was Synthetic Minority Over-sampling Technique (SMOTE), which generates synthetic samples of the minority class by interpolating between existing samples.

The oversampling improved the performance of the model significantly, as indicated by the higher accuracy and r2_score on the random train-test split

## Results
The Random Forest algorithm performs very well on the oversampled wine quality dataset, with an accuracy of 0.95 and r2_score of 0.96 on a random train test split. The most important attributes for predicting wine quality are alcohol, volatile acidity, and sulphates, which have the highest feature importances in the model. The analysis also shows that there is a negative correlation between volatile acidity and wine quality, while there is a positive correlation between alcohol and wine quality.

##Streamlit App
This repository contains a concise Streamlit application for wine quality analysis. Using a machine learning model, the app predicts the quality of wines based on their chemical properties. Users can input the wine features, and the app provides quality predictions and corresponding probabilities. To run the app locally, follow the instructions in the README file. 

## Libraries
This project utilizes several Python libraries for data analysis and machine learning, including:

pandas: a library for data manipulation and analysis, used to read in and process the wine quality dataset.\
numpy: a library for numerical operations, used for mathematical computations in data preprocessing and model training.\
matplotlib: a library for data visualization, used to create histograms, boxplots, scatterplots, and other visualizations for EDA.\
seaborn: a library for statistical data visualization, used to create correlation matrices and heatmaps for exploring relationships between attributes.\
scikit-learn: a library for machine learning, used to perform GridSearch, Cross Validation, and implement the Random Forest algorithm for wine quality prediction.\
imblearn: A library which is used for oversampling to address the class imbalance in the wine quality dataset. This library provides various techniques for oversampling, including SMOTE (Synthetic Minority Over-sampling Technique) and ADASYN (Adaptive Synthetic Sampling).

## Conclusion
This project demonstrates the use of EDA, GridSearch, and Random Forest algorithms in predicting wine quality based on physical and chemical characteristics. The Random Forest model performs well on the wine quality dataset and can be used to predict the quality of a wine with reasonable accuracy.

In addition to the machine learning techniques, oversampling was performed using the imblearn library, which improved the accuracy of the model. The final model achieved an accuracy of 0.95 and an r2 score of 0.96 on a random train-test split.

Further analysis could include exploring other machine learning algorithms or analyzing the wine quality dataset separately for red wines.

##How to Run the project ?

To run a Streamlit app using the code provided in the repository's README file, follow these steps:

1) Clone the Repository: Start by cloning the repository to your local machine. You can use the following command in your terminal or command prompt:
```python
git clone <repository-url>
```
2) Navigate to the Repository: Change your working directory to the cloned repository:
```python
cd <repository-folder>
```
3) Set Up a Virtual Environment (Optional): It is recommended to create a virtual environment to isolate the app's dependencies. You can use tools like venv or conda to create a virtual environment:
```python
conda create --name myenv
conda activate myenv
```
4) Install Dependencies: Install the required dependencies by running the following command, assuming you have pip installed:

```python
pip install -r requirements.txt
```
This command will install all the necessary packages specified in the requirements.txt file.

5) Run the Streamlit App: Once the dependencies are installed, you can start the Streamlit app by executing the following command:

```python
streamlit run app.py
```
This command will launch the Streamlit development server and start the app.

7) Access the App: After running the above command, you will see a local web address (e.g., http://localhost:8501). Open a web browser and enter the provided address to access the Streamlit app.

You should now be able to interact with the Streamlit app and perform wine quality analysis based on the provided instructions and features.
