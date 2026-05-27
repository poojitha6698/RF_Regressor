import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    # Check Missing Values
    print(df.isnull().sum())

    # Label Encoding
    le = LabelEncoder()

    categorical_cols = ['sex', 'smoker', 'region']

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df