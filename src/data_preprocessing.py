def preprocess_data(df):

    # Drop unwanted columns
    df.drop(['id', 'date'], axis=1, inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    return df