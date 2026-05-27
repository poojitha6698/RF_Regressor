from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(X_train, y_train):

    rf = RandomForestRegressor(random_state=42)

    # Hyperparameters
    params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='r2'
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)

    os.makedirs('save_models', exist_ok=True)

    # Save Model
    joblib.dump(best_model,
                'save_models/random_forest_model.pkl')

    return best_model