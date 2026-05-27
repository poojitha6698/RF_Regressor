from sklearn.preprocessing import StandardScaler


def feature_engineering(df):

    X = df.drop('price', axis=1)
    y = df['price']

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler