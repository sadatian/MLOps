import subprocess
import os
import pytest
import pandas as pd

def test_mlops_cli_help():
    """Verify that the mlops help menu is outputted successfully."""
    result = subprocess.run(["mlops", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Unified Command Line Tool" in result.stdout

def test_mlops_status():
    """Verify that the status diagnostics command works."""
    result = subprocess.run(["mlops", "status"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Python Executable:" in result.stdout
    assert "OS Platform:" in result.stdout

def test_mlops_baseline():
    """Verify the heuristic baseline run command executes and outputs correct pricing predictions."""
    input_path = "tests/test_raw.csv"
    output_path = "tests/test_predictions.csv"
    
    # Setup test file
    df = pd.DataFrame({"area_sqft": [1000, 2000], "bedrooms": [2, 3]})
    df.to_csv(input_path, index=False)
    
    try:
        result = subprocess.run([
            "mlops", "baseline", "run",
            "--input", input_path,
            "--output", output_path
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Heuristic inference completed" in result.stdout
        
        # Verify output file
        assert os.path.exists(output_path)
        out_df = pd.read_csv(output_path)
        assert "heuristic_price" in out_df.columns
        # Price: 150000 + 1000 * 150 = 300000
        assert out_df.iloc[0]["heuristic_price"] == 300000.0
    finally:
        # Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

def test_mlops_gate():
    """Verify that the quality gating subcommand handles thresholds and passes/fails appropriately."""
    metrics_path = "tests/test_metrics.json"
    
    # 1. Gate passes
    import json
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": 0.85}, f)
        
    try:
        result = subprocess.run([
            "mlops", "gate", "check",
            "--metrics", metrics_path,
            "--threshold", "0.80"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Quality Gate Passed!" in result.stdout
        
        # 2. Gate fails
        result_fail = subprocess.run([
            "mlops", "gate", "check",
            "--metrics", metrics_path,
            "--threshold", "0.90"
        ], capture_output=True, text=True)
        assert result_fail.returncode == 1
        assert "Quality Gate Failed!" in result_fail.stdout
    finally:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)

@pytest.mark.parametrize("cmd", [
    "docs", "moto", "dvc", "mlflow", "pipeline", "serve",
    "container", "monitor", "ci", "orchestrate", "feature",
    "serve-grpc", "predict-batch", "ct", "llm", "iac"
])
def test_subcommand_helps(cmd):
    """Verify help outputs of all other unified subcommands."""
    result = subprocess.run(["mlops", cmd, "--help"], capture_output=True, text=True)
    assert result.returncode == 0

def test_mlops_cli_local_flag():
    """Verify that the --local flag is correctly intercepted and runs status locally."""
    env = os.environ.copy()
    env["MLOPS_LOCAL"] = "1"
    result = subprocess.run(["mlops", "status", "--local"], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Python Executable:" in result.stdout

def test_mlops_cli_local_env():
    """Verify that MLOPS_LOCAL=1 environment variable executes locally."""
    env = os.environ.copy()
    env["MLOPS_LOCAL"] = "1"
    result = subprocess.run(["mlops", "status"], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Python Executable:" in result.stdout

def test_internal_helpers():
    """Test find_project_root and should_run_local helper logic directly."""
    from mlops.cli import find_project_root, should_run_local
    
    root = find_project_root()
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    assert os.path.exists(os.path.join(root, "Dockerfile"))

    # If we set MLOPS_LOCAL=1, should_run_local should return True
    os.environ["MLOPS_LOCAL"] = "1"
    assert should_run_local() is True
    
    # Clean up
    del os.environ["MLOPS_LOCAL"]


def test_mlops_init():
    """Verify that the init command with --local-dev executes successfully."""
    # Run in a mock environment using MLOPS_LOCAL=1
    env = os.environ.copy()
    env["MLOPS_LOCAL"] = "1"
    
    result = subprocess.run([
        "mlops", "init", "--local-dev"
    ], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Initializing local development environment..." in result.stdout
    assert "MLflow tracking setup successfully." in result.stdout


def test_mlops_predict():
    """Verify that the predict command successfully executes local inference."""
    env = os.environ.copy()
    env["MLOPS_LOCAL"] = "1"
    
    result = subprocess.run([
        "mlops", "predict", "heuristic",
        "--feature", '{"area_sqft": 1500.0, "bedrooms": 3}'
    ], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert '"predicted_price_usd": 375000.0' in result.stdout

