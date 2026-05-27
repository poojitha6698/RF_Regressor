from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate_model(model, X_test, y_test, feature_names):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Feature Importance
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    feature_df = feature_df.sort_values(
        by='Importance',
        ascending=False
    )

    os.makedirs('plots', exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.barh(feature_df['Feature'],
             feature_df['Importance'])

    plt.title('Feature Importance')

    plt.savefig('plots/feature_importance.png')
    plt.close()