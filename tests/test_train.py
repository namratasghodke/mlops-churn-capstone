import pytest
from unittest import mock
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import sys
import os

# Add train.py path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/training")))
import train  # Import your train.py module

# -------------------------
# Fixtures for mocking MLflow
# -------------------------
@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "start_run", mock.MagicMock())
    monkeypatch.setattr(mlflow, "log_param", mock.MagicMock())
    monkeypatch.setattr(mlflow, "log_params", mock.MagicMock())
    monkeypatch.setattr(mlflow, "log_metric", mock.MagicMock())
    monkeypatch.setattr(mlflow, "log_metrics", mock.MagicMock())
    monkeypatch.setattr(mlflow.sklearn, "log_model", mock.MagicMock())

# -------------------------
# Test if model trains correctly with sample data
# -------------------------
def test_model_training():
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)

# -------------------------
# Test metrics computation
# -------------------------
def test_metrics_computation():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    assert acc == 0.75
    assert round(f1, 2) == 0.67

# -------------------------
# Test MLflow logging calls
# -------------------------
def test_mlflow_logging_calls(monkeypatch):
    mock_run = mock.MagicMock()
    monkeypatch.setattr(mlflow, "start_run", mock.MagicMock(return_value=mock_run))

    with mlflow.start_run():
        mlflow.log_params({"n_estimators": 100})
        mlflow.log_metrics({"accuracy": 0.8})

    mlflow.start_run.assert_called_once()
    mlflow.log_params.assert_called_once_with({"n_estimators": 100})
    mlflow.log_metrics.assert_called_once_with({"accuracy": 0.8})

# -------------------------
# Optional: Test train.py main function (if refactored)
# -------------------------
#def test_train_main_runs(monkeypatch):
    # Mock reading CSVs to avoid real files
    #mock_X = pd.DataFrame({'f1': [1,2], 'f2':[3,4]})
   # mock_y = pd.DataFrame({'target': [0,1]})
    
  #  monkeypatch.setattr(pd, "read_csv", lambda filepath: mock_X if "X" in filepath else mock_y)
    
    # Run train.py code in a way that avoids MLflow actually logging
   # try:
   #     train.main()  # You need to refactor train.py to have a main() function
 #   except Exception as e:
  #      pytest.fail(f"train.main() failed: {e}")
