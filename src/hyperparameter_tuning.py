from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def tune_hyperparameters(X_train, y_train):

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print('Best Parameters:', grid_search.best_params_)

    return grid_search.best_estimator_