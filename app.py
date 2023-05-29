import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

st.write('''
    <h1 style="text-align:center;">Wine Quality Prediction App</h1>
    <p style="text-align:center;">This app is used to predict the quality of wines with respect to its chemical composition</p>
''', unsafe_allow_html=True)

# Load the wine quality dataset
wine_data = pd.read_csv('.\Assets\cleaned_data.csv')

class QualityPredictionApp:
    def input():
        # Split the data into features (X) and target (y)
        X = wine_data.drop('quality', axis=1)
        y = wine_data['quality']

        col1,col2,col3=st.columns(3)
        # Create input widgets for user input
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", float(X['fixed acidity'].min()), float(X['fixed acidity'].max()))
        with col2:
            volatile_acidity = st.number_input("Volatile Acidity", float(X['volatile acidity'].min()), float(X['volatile acidity'].max()))
        with col3:
            citric_acid = st.number_input("Citric Acid", float(X['citric acid'].min()), float(X['citric acid'].max()))
        
        col1,col2,col3=st.columns(3)
        with col1:
            residual_sugar = st.number_input("Residual Sugar", float(X['residual sugar'].min()), float(X['residual sugar'].max()))
        with col2:
            chlorides = st.number_input("Chlorides", float(X['chlorides'].min()), float(X['chlorides'].max()))
        with col3:
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", float(X['free sulfur dioxide'].min()), float(X['free sulfur dioxide'].max()))
        
        
        col1,col2,col3=st.columns(3)
        with col1:
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", float(X['total sulfur dioxide'].min()), float(X['total sulfur dioxide'].max()))
        with col2:
            density = st.number_input("Density", float(X['density'].min()), float(X['density'].max()))
        with col3:
            pH = st.number_input("pH", float(X['pH'].min()), float(X['pH'].max()))
        
        col1,col2,col3=st.columns(3)
        with col1:
            sulphates = st.number_input("Sulphates", float(X['sulphates'].min()), float(X['sulphates'].max()))
        with col2:
            alcohol = st.number_input("Alcohol", float(X['alcohol'].min()), float(X['alcohol'].max()))
        
        data={
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        }
        features=pd.DataFrame(data,index=[0])
        return features

    def predict_data(data):
        st.write('Getting Predictions')
        return PredictPipeline.initiate_prediction(PredictPipeline(),data)

if __name__ == '__main__':
    features=QualityPredictionApp.input()

    # Create a style for the button
    style = """
    <style>
    .stButton {
    size=100px;
    background-color: #4F8BF9;
    color: white;
    height: 35px;
    border-radius:75%;
    width: 30px;
    text-align: center;
    }
    </style>
    """

    # Add the style to the page
    st.markdown(style, unsafe_allow_html=True)

    # Create the button
    button = st.button("Predict")
    if button:
        st.empty()
        print('Button Pressed')
        model_pred,model_prob=QualityPredictionApp.predict_data(features)
        st.write('Prediction')
        st.write(model_pred)

        st.write('Prediction Probability')
        st.write(model_prob)
