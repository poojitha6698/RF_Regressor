import joblib


def save_model(model, scaler):

    joblib.dump(model, 'save_models/random_forest_model.pkl')

    joblib.dump(scaler, 'save_models/scaler.pkl')

    print('Model Saved Successfully')