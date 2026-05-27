from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.hyperparameter_tuning import tune_hyperparameters
from src.evaluation import evaluate_model
from src.save_model import save_model

# Load Dataset

df = load_dataset('data/kc_house_data.csv')

# Preprocessing

df = preprocess_data(df)

# Feature Engineering

X_scaled, y, scaler = feature_engineering(df)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Hyperparameter Tuning

best_model = tune_hyperparameters(X_train, y_train)

# Evaluation

evaluate_model(best_model, X_test, y_test)

# Save Model

save_model(best_model, scaler)