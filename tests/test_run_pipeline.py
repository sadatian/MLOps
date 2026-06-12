import os
import json
import pytest
import pandas as pd
import numpy as np
import mlflow

from src.data_experimentation.module_06_integrated_pipeline.run_pipeline import prepare_data, train_model, evaluate_model

def test_pipeline_execution(tmp_path):
    # Setup temporary paths
    raw_path = tmp_path / "raw.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    # Set temporary MLflow tracking directory to prevent polluting local runs
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_tracking_dir = tmp_path / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")

    # Generate small dummy dataset
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        "area_sqft": np.random.randint(800, 3000, size=20),
        "bedrooms": np.random.randint(1, 5, size=20),
        "price_usd": np.random.randint(150000, 600000, size=20)
    })
    dummy_data.to_csv(raw_path, index=False)

    # 1. Test data preparation stage
    prepare_data(str(raw_path), str(train_path), str(test_path))
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # Load and check preprocessed columns
    train_df = pd.read_csv(train_path)
    assert "area_k_sqft" in train_df.columns

    # 2. Test model training stage
    model = train_model(str(train_path), str(model_path), n_estimators=5, max_depth=2)
    assert os.path.exists(model_path)
    assert model is not None

    # 3. Test evaluation stage
    evaluate_model(str(test_path), str(model_path), str(metrics_path))
    assert os.path.exists(metrics_path)

    # Check metric contents
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "rmse" in metrics
    assert "r2_score" in metrics
