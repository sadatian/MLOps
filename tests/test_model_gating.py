import os
import json
import pytest
from src.module_10_ci_ml_automation.ci_ml_guide import verify_model_performance

def test_verify_model_performance_success(tmp_path):
    # Setup temporary metrics file with passing values
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"rmse": 180000.0, "r2_score": 0.45}, f)
        
    # Check that it passes the gate (returns True)
    assert verify_model_performance(str(metrics_path), max_rmse_allowed=200000.0, min_r2_required=0.4) is True

def test_verify_model_performance_fails_rmse(tmp_path):
    # Setup temporary metrics file where RMSE is too high
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"rmse": 220000.0, "r2_score": 0.45}, f)
        
    # Check that it fails the gate (returns False)
    assert verify_model_performance(str(metrics_path), max_rmse_allowed=200000.0, min_r2_required=0.4) is False

def test_verify_model_performance_fails_r2(tmp_path):
    # Setup temporary metrics file where R2 is too low
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"rmse": 180000.0, "r2_score": 0.25}, f)
        
    # Check that it fails the gate (returns False)
    assert verify_model_performance(str(metrics_path), max_rmse_allowed=200000.0, min_r2_required=0.4) is False

def test_verify_model_performance_missing_file():
    # Verify that a FileNotFoundError is raised for a non-existent file path
    with pytest.raises(FileNotFoundError):
        verify_model_performance("non_existent_metrics_file.json")
