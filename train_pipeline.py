from src.data_preprocessing import load_data, preprocess_data
from src.eda import perform_eda
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.model_evaluation import evaluate_model

from src.utils import (
    create_folder,
    dataset_info,
    split_features_target
)

from sklearn.model_selection import train_test_split

# Create Required Folders
create_folder('plots')
create_folder('save_models')

# Load Dataset
path = 'data/insurance.csv'

df = load_data(path)

# Dataset Information
dataset_info(df)

# Preprocessing
df = preprocess_data(df)

# EDA
perform_eda(df)

# Features and Target
X, y = split_features_target(df, 'charges')

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Engineering
X_train_scaled, X_test_scaled, scaler = feature_engineering(
    X_train,
    X_test
)

# Train Model
model = train_model(X_train_scaled, y_train)

# Evaluate Model
evaluate_model(
    model,
    X_test_scaled,
    y_test,
    X.columns
)