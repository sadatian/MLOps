import boto3
import pickle
from moto import mock_aws

def test_s3_model_registry_simulation():
    bucket_name = "mlops-model-registry"
    object_key = "models/housing_model.pkl"
    
    sample_model = {
        "model_name": "housing_price_predictor",
        "features": ["area_k_sqft", "bedrooms"],
        "weights": [120.5, 45000.0],
        "intercept": 50000.0
    }
    serialized_model = pickle.dumps(sample_model)
    
    with mock_aws():
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket=bucket_name)
        
        # Test creation check
        buckets = s3_client.list_buckets()["Buckets"]
        assert any(b["Name"] == bucket_name for b in buckets)
        
        # Upload model
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=serialized_model
        )
        
        # Download and verify
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        retrieved_bytes = response["Body"].read()
        retrieved_model = pickle.loads(retrieved_bytes)
        
        assert retrieved_model["model_name"] == "housing_price_predictor"
        assert retrieved_model["features"] == ["area_k_sqft", "bedrooms"]
        assert retrieved_model["weights"] == [120.5, 45000.0]
        assert retrieved_model["intercept"] == 50000.0
