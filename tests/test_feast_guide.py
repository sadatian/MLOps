import pandas as pd
import boto3
from moto import mock_aws
from src.advanced_mlops.module_13_feature_store.feast_guide import SimulatedFeatureStore

def test_feature_store_simulation():
    credit_scores_history = pd.DataFrame([
        {"user_id": 1001, "timestamp": "2026-01-01 00:00:00", "credit_score": 650},
        {"user_id": 1001, "timestamp": "2026-03-01 00:00:00", "credit_score": 670},
    ])
    
    transaction_volume_history = pd.DataFrame([
        {"user_id": 1001, "timestamp": "2026-01-01 00:00:00", "transaction_volume_30d": 1200.50},
    ])
    
    loan_applications = pd.DataFrame([
        {"user_id": 1001, "timestamp": "2026-03-15 12:00:00", "action": "loan_request"},
    ])
    
    with mock_aws():
        bucket = "mlops-feature-store-test"
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        
        fs = SimulatedFeatureStore(s3_bucket=bucket)
        fs.upload_offline_features("credit_scores", credit_scores_history)
        fs.upload_offline_features("transaction_volumes", transaction_volume_history)
        
        # Test historical feature retrieval (ASOF join)
        hist_df = fs.get_historical_features(
            entity_df=loan_applications,
            feature_names=["credit_scores", "transaction_volumes"]
        )
        
        row = hist_df.iloc[0]
        assert row["credit_score"] == 670
        assert row["transaction_volume_30d"] == 1200.50
        
        # Test materialization
        fs.materialize(end_timestamp="2026-03-15 00:00:00")
        
        # Test online retrieval
        online = fs.get_online_features(
            entity_keys=[{"user_id": 1001}],
            feature_names=["credit_score", "transaction_volume_30d"]
        )
        assert len(online) == 1
        assert online[0]["credit_score"] == 670
        assert online[0]["transaction_volume_30d"] == 1200.50
