import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import (
    load_data,
    preprocess_data
)

# Page Config
st.set_page_config(
    page_title='Insurance Charges Prediction',
    layout='wide'
)

st.title('💰 Insurance Charges Prediction')

# Load Dataset
path = 'data/insurance.csv'

df = load_data(path)

df = preprocess_data(df)

# Load Model
model = joblib.load(
    'save_models/random_forest_model.pkl'
)

# Sidebar Inputs
st.sidebar.header('User Input Features')

age = st.sidebar.slider('Age', 18, 100, 30)

sex = st.sidebar.selectbox(
    'Sex',
    ['male', 'female']
)

bmi = st.sidebar.slider(
    'BMI',
    10.0,
    50.0,
    25.0
)

children = st.sidebar.slider(
    'Children',
    0,
    10,
    1
)

smoker = st.sidebar.selectbox(
    'Smoker',
    ['yes', 'no']
)

region = st.sidebar.selectbox(
    'Region',
    ['southwest', 'southeast',
     'northwest', 'northeast']
)

# Encoding
sex = 1 if sex == 'male' else 0
smoker = 1 if smoker == 'yes' else 0

region_dict = {
    'southwest':0,
    'southeast':1,
    'northwest':2,
    'northeast':3
}

region = region_dict[region]

# Input Data
input_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'bmi':[bmi],
    'children':[children],
    'smoker':[smoker],
    'region':[region]
})

# Prediction
prediction = model.predict(input_data)

st.subheader('Predicted Insurance Charges')

st.success(f'${prediction[0]:,.2f}')

# Dataset Preview
st.subheader('Dataset Preview')

st.dataframe(df.head())

# EDA Visualizations
st.subheader('EDA Visualizations')

st.image('plots/charges_distribution.png')

st.image('plots/bmi_distribution.png')

st.image('plots/correlation_heatmap.png')

st.image('plots/age_vs_charges.png')

st.image('plots/feature_importance.png')