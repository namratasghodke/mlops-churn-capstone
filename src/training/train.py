import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# --- Set base directory relative to this script ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# --- Load processed data ---
X_train = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/X_train.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/y_train.csv'))
X_val = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/X_val.csv'))
y_val = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/y_val.csv'))

# --- Set experiment ---
mlflow.set_experiment("churn-prediction")

# --- Define and train model ---
params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
model = RandomForestClassifier(**params)
model.fit(X_train, y_train.values.ravel())

# --- Evaluate model ---
preds = model.predict(X_val)
metrics = {
    "accuracy": accuracy_score(y_val, preds),
    "f1_score": f1_score(y_val, preds)
}

# --- Log to MLflow ---
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnModel")

print("✅ Training completed successfully.")
