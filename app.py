import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title='Random Forest Regression App',
    layout='wide'
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv('data/kc_house_data.csv')

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------

model = joblib.load(
    'save_models/random_forest_model.pkl'
)

scaler = joblib.load(
    'save_models/scaler.pkl'
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title('House Price Prediction using Random Forest Regression')

st.write('Machine Learning Web App using Streamlit')

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header('Project Information')

st.sidebar.write("""
This project predicts house prices using:
- Random Forest Regression
- Hyperparameter Tuning
- Feature Scaling
- EDA Visualizations
- Feature Importance
""")

# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------

st.header('Exploratory Data Analysis')

# Numeric Data Only

numeric_df = df.select_dtypes(include=['number'])

# --------------------------------------------------
# PRICE DISTRIBUTION
# --------------------------------------------------

st.subheader('Price Distribution')

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.histplot(
    df['price'],
    kde=True,
    ax=ax1
)

ax1.set_title('Price Distribution')

st.pyplot(fig1)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader('Correlation Heatmap')

fig2, ax2 = plt.subplots(figsize=(15,10))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm',
    ax=ax2
)

ax2.set_title('Correlation Heatmap')

st.pyplot(fig2)

# --------------------------------------------------
# SCATTER PLOT
# --------------------------------------------------

st.subheader('sqft_living vs Price')

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.scatterplot(
    x=df['sqft_living'],
    y=df['price'],
    ax=ax3
)

ax3.set_title('sqft_living vs Price')

st.pyplot(fig3)

# --------------------------------------------------
# BOXPLOT
# --------------------------------------------------

st.subheader('Bedrooms Boxplot')

fig4, ax4 = plt.subplots(figsize=(8,5))

sns.boxplot(
    x=df['bedrooms'],
    ax=ax4
)

ax4.set_title('Bedrooms Boxplot')

st.pyplot(fig4)

# --------------------------------------------------
# USER INPUT SECTION
# --------------------------------------------------

st.header('Enter House Details')

col1, col2 = st.columns(2)

with col1:

    bedrooms = st.number_input(
        'Bedrooms',
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        'Bathrooms',
        min_value=1.0,
        max_value=10.0,
        value=2.0
    )

    sqft_living = st.number_input(
        'sqft_living',
        value=1500
    )

    sqft_lot = st.number_input(
        'sqft_lot',
        value=5000
    )

    floors = st.number_input(
        'floors',
        value=1.0
    )

    waterfront = st.number_input(
        'waterfront',
        value=0
    )

    view = st.number_input(
        'view',
        value=0
    )

    condition = st.number_input(
        'condition',
        value=3
    )

    grade = st.number_input(
        'grade',
        value=7
    )

with col2:

    sqft_above = st.number_input(
        'sqft_above',
        value=1500
    )

    sqft_basement = st.number_input(
        'sqft_basement',
        value=0
    )

    yr_built = st.number_input(
        'yr_built',
        value=1990
    )

    yr_renovated = st.number_input(
        'yr_renovated',
        value=0
    )

    zipcode = st.number_input(
        'zipcode',
        value=98001
    )

    lat = st.number_input(
        'lat',
        value=47.5
    )

    long = st.number_input(
        'long',
        value=-122.2
    )

    sqft_living15 = st.number_input(
        'sqft_living15',
        value=1500
    )

    sqft_lot15 = st.number_input(
        'sqft_lot15',
        value=5000
    )

# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'waterfront': [waterfront],
    'view': [view],
    'condition': [condition],
    'grade': [grade],
    'sqft_above': [sqft_above],
    'sqft_basement': [sqft_basement],
    'yr_built': [yr_built],
    'yr_renovated': [yr_renovated],
    'zipcode': [zipcode],
    'lat': [lat],
    'long': [long],
    'sqft_living15': [sqft_living15],
    'sqft_lot15': [sqft_lot15]
})

# --------------------------------------------------
# SCALE INPUT DATA
# --------------------------------------------------

input_scaled = scaler.transform(input_data)

# --------------------------------------------------
# PREDICTION SECTION
# --------------------------------------------------

st.header('Prediction')

if st.button('Predict House Price'):

    prediction = model.predict(input_scaled)

    st.success(
        f'Predicted House Price: ${prediction[0]:,.2f}'
    )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.header('Model Evaluation Metrics')

# Features and Target

X = df.drop(
    ['price', 'id', 'date'],
    axis=1
)

y = df['price']

# Scale Features

X_scaled = scaler.transform(X)

# Predictions

y_pred = model.predict(X_scaled)

# Metrics

mae = mean_absolute_error(y, y_pred)

mse = mean_squared_error(y, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y, y_pred)

# --------------------------------------------------
# DISPLAY METRICS
# --------------------------------------------------

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    'MAE',
    f'{mae:.2f}'
)

metric2.metric(
    'MSE',
    f'{mse:.2f}'
)

metric3.metric(
    'RMSE',
    f'{rmse:.2f}'
)

metric4.metric(
    'R2 Score',
    f'{r2:.4f}'
)

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.header('Actual vs Predicted Prices')

fig5, ax5 = plt.subplots(figsize=(8,6))

ax5.scatter(y, y_pred)

ax5.set_xlabel('Actual Prices')

ax5.set_ylabel('Predicted Prices')

ax5.set_title('Actual vs Predicted Prices')

st.pyplot(fig5)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.header('Feature Importance')

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

fig6, ax6 = plt.subplots(figsize=(12,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df,
    ax=ax6
)

ax6.set_title('Feature Importance')

st.pyplot(fig6)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.header('Dataset Preview')

st.dataframe(df.head())

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header('Dataset Information')

st.write(f'Shape of Dataset: {df.shape}')

st.write('Columns:')

st.write(df.columns.tolist())

# --------------------------------------------------
# PROJECT SUMMARY
# --------------------------------------------------

st.header('Project Summary')

st.write("""
This project predicts house prices using
Random Forest Regression.

Features Included:
- EDA Visualizations
- Feature Scaling
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance
- Prediction System
- Streamlit Deployment
""")