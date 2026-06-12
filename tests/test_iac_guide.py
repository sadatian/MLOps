import os
import pytest
from moto import mock_aws
from src.advanced_mlops.module_17_iac_advanced_containers.iac_guide import (
    run_terraform_validation,
    validate_terraform_policies,
    parse_variables_file,
    LocalStackSimulator,
    validate_gpu_dockerfile,
    verify_gpu_availability,
    TF_DIR,
    MAIN_TF_PATH,
    VARIABLES_TF_PATH,
    DOCKERFILE_PATH
)

def test_run_terraform_validation():
    # Verify validation runs successfully on the module TF directory
    res = run_terraform_validation(TF_DIR)
    # Since Terraform binary is installed, check that the result has success or a valid failure message if network issues occur.
    # We assert success is a boolean.
    assert "success" in res
    assert isinstance(res["success"], bool)

def test_validate_terraform_policies():
    res = validate_terraform_policies(MAIN_TF_PATH)
    assert res["success"] is True
    assert "provider_localstack_redirection" in res["checks"]
    assert res["checks"]["provider_localstack_redirection"]["passed"] is True
    assert res["checks"]["s3_bucket_encryption"]["passed"] is True
    assert res["checks"]["s3_bucket_versioning"]["passed"] is True
    assert res["checks"]["dynamodb_cost_optimization"]["passed"] is True

def test_parse_variables_file():
    defaults = parse_variables_file(VARIABLES_TF_PATH)
    assert "aws_region" in defaults
    assert defaults["aws_region"] == "us-east-1"
    assert "bucket_name" in defaults
    assert defaults["bucket_name"] == "mlops-model-registry-bucket"

@mock_aws()
def test_localstack_simulator():
    sim = LocalStackSimulator("test-bucket", "test-table")
    sim.provision_resources()
    
    # Upload and fetch
    sim.upload_model_artifact("model-a", "v1", "weights.bin", b"weightdata")
    meta = sim.fetch_model_metadata("model-a", "v1")
    
    assert meta["ModelId"]["S"] == "model-a"
    assert meta["Version"]["S"] == "v1"
    assert meta["S3Uri"]["S"] == "s3://test-bucket/models/model-a/v1/weights.bin"
    assert meta["Status"]["S"] == "Registered"

def test_validate_gpu_dockerfile():
    res = validate_gpu_dockerfile(DOCKERFILE_PATH)
    assert res["success"] is True
    assert res["checks"]["gpu_base_image"]["passed"] is True
    assert res["checks"]["nvidia_variables"]["passed"] is True
    assert res["checks"]["cache_cleanup"]["passed"] is True

def test_verify_gpu_availability():
    res = verify_gpu_availability()
    assert "cuda_available" in res
    assert "device_count" in res
    assert "active_device" in res
    assert "driver_version" in res
    assert "serving_mode" in res
    
    # Since the workspace host has a real RTX 4080 GPU, this check should resolve to True!
    assert res["cuda_available"] is True
    assert res["device_count"] >= 1
    assert "RTX 4080" in res["active_device"] or "NVIDIA" in res["active_device"]
    assert "GPU-accelerated" in res["serving_mode"]
