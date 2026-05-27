import os
import joblib
import pandas as pd

# Create Folder
def create_folder(folder_name):

    os.makedirs(folder_name, exist_ok=True)

    print(f"{folder_name} folder created successfully")


# Save Model
def save_model(model, path):

    joblib.dump(model, path)

    print(f"Model saved at {path}")


# Load Model
def load_model(path):

    model = joblib.load(path)

    print("Model loaded successfully")

    return model


# Display Dataset Information
def dataset_info(df):

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Columns:")
    print(df.columns)

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())


# Export DataFrame
def export_dataframe(df, path):

    df.to_csv(path, index=False)

    print(f"Data exported to {path}")


# Check Null Values
def check_null_values(df):

    null_values = df.isnull().sum()

    print("\nMissing Values:")
    print(null_values)

    return null_values


# Feature and Target Separation
def split_features_target(df, target_column):

    X = df.drop(target_column, axis=1)

    y = df[target_column]

    return X, y