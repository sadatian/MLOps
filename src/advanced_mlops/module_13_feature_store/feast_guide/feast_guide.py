# %% [markdown]
# # 🗄️ Feature Store Implementation (Simulated Feast)
#
# In production MLOps, features are consumed by two distinct environments:
# 1. **Offline Training:** Requires historical feature values at the exact time a past event occurred to train models without data leakage.
# 2. **Online Inference:** Requires low-latency, real-time retrieval of the latest feature value for a specific entity (e.g., user_id) to serve predictions.
#
# ### The Feature Store Dual-Database Architecture
#
# A **Feature Store** solves this training-serving skew by maintaining a dual-database architecture under a single registry interface:
# *   **Offline Store:** A historical storage layout (AWS S3, Snowflake, BigQuery) containing parquet files of features over time. This layer uses temporal joins (ASOF Joins) to align features with target labels without looking into the future (preventing data leakage).
# *   **Online Store:** A low-latency database (Redis, DynamoDB, SQLite) storing only the latest feature values. The process of syncing historical records to the low-latency store is called **Materialization**.
#
# ```mermaid
#  graph TD
#      subgraph raw_ingestion ["Raw Ingestion"]
#          A["Production Database"] -->|"Batch Export"| B["Parquet Logs S3"]
#      end
#
#      subgraph offline_feature_layer_historical_analysis ["Offline Feature Layer (Historical Analysis)"]
#          C["Offline Store: AWS S3 / Parquet"]
#          D["Label Dataframe: user_id & event_timestamp"] -->|"get_historical_features"| E["Temporal ASOF Join"]
#          C --> E
#          E -->|"Prevent Data Leakage"| F["Clean Training Dataset"]
#      end
#
#      subgraph online_feature_layer_low_latency_serving ["Online Feature Layer (Low-latency Serving)"]
#          G["Materialization Daemon"]
#          H["(Online Store: Redis / DynamoDB / SQLite)"]
#          I["Real-Time API Inference Request: user_id"] -->|"get_online_features"| J["O1 SQLite Key-Value Lookup"]
#          H --> J
#          J -->|"Fetch Features"| K["FastAPI Prediction Endpoint"]
#      end
#
#      B --> C
#      C -->|"materialize end_timestamp"| G
#      G -->|"Write latest entity state"| H
#
#      B ~~~ C
#      F ~~~ G
#      F ~~~ I
#
#      style F fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#      style H fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#      style K fill:#bbdefb,stroke:#1976d2,stroke-width:1.5px
#
# ```
#
# In this module, we will build a simulated feature store to explore:
# 1. Simulating an Offline Store in a **mock S3 bucket** via `boto3` and `moto`.
# 2. Preventing **Data Leakage** using a temporal **ASOF Join** (`pd.merge_asof`) for point-in-time correctness.
# 3. **Materialization** (synchronizing S3 offline historical features to a low-latency SQLite online database).
# 4. **Low-Latency Online Retrieval** of features for online model inference.


# %%
# Ensure we run from the project root directory
import os
import sys

# Locate project root (searching upwards for mkdocs.yml)
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
        break
    current_dir = os.path.dirname(current_dir)

if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
    os.chdir(current_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
else:
    print("⚠️ Could not find project root containing mkdocs.yml.")

import io
import time
import sqlite3
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import boto3
from moto import mock_aws

# Ensure boto3 uses dummy credentials for local simulation
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# %% [markdown]
# ## 🗄️ 1. Defining the FeatureStore Simulator
# We will design a class `SimulatedFeatureStore` modeling standard Feast APIs:
# - `get_historical_features()` for point-in-time training joins.
# - `materialize()` to sync features from S3 to SQLite.
# - `get_online_features()` for low-latency production serving.

# %%
class SimulatedFeatureStore:
    def __init__(self, s3_bucket: str):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client("s3", region_name="us-east-1")
        
        # Initialize Online Store (SQLite In-Memory Database)
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self._setup_online_db()

    def _setup_online_db(self):
        """Creates the online feature tables."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS online_user_features (
                user_id INTEGER PRIMARY KEY,
                credit_score INTEGER,
                transaction_volume_30d REAL,
                last_updated TIMESTAMP
            )
        """)
        self.conn.commit()

    def upload_offline_features(self, feature_name: str, df: pd.DataFrame):
        """Uploads historical features to the mock S3 offline store as Parquet."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        
        s3_key = f"offline/{feature_name}.parquet"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=buffer.getvalue()
        )
        print(f"📁 Offline Store: Uploaded '{feature_name}' historical records to s3://{self.s3_bucket}/{s3_key}")

    def read_offline_features(self, feature_name: str) -> pd.DataFrame:
        """Downloads historical features from the mock S3 offline store."""
        s3_key = f"offline/{feature_name}.parquet"
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        parquet_data = response["Body"].read()
        return pd.read_parquet(io.BytesIO(parquet_data))

    def get_historical_features(
        self, 
        entity_df: pd.DataFrame, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Executes a point-in-time ASOF join to associate each entity event with
        the latest feature values recorded PRIOR OR AT the event timestamp.
        Prevents data leakage.
        """
        # Ensure event timestamp is datetime and sorted
        result_df = entity_df.copy()
        result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
        result_df = result_df.sort_values("timestamp")
        
        for feature_name in feature_names:
            # Download feature dataframe from S3 offline store
            feature_df = self.read_offline_features(feature_name)
            feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"])
            feature_df = feature_df.sort_values("timestamp")
            
            # Perform temporal (ASOF) join
            # Matches keys where feature_df.timestamp <= result_df.timestamp
            result_df = pd.merge_asof(
                result_df,
                feature_df,
                on="timestamp",
                by="user_id",
                direction="backward"
            )
            
        return result_df

    def materialize(self, end_timestamp: str):
        """
        Materializes (synchronizes) offline features to the online store.
        Finds the latest value for each entity up to `end_timestamp`.
        """
        end_dt = pd.to_datetime(end_timestamp)
        print(f"🔄 Materializing features to Online Store up to: {end_timestamp}...")
        
        # Read from mock S3
        credit_df = self.read_offline_features("credit_scores")
        tx_df = self.read_offline_features("transaction_volumes")
        
        credit_df["timestamp"] = pd.to_datetime(credit_df["timestamp"])
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])
        
        # Filter up to cutoff time
        credit_df = credit_df[credit_df["timestamp"] <= end_dt]
        tx_df = tx_df[tx_df["timestamp"] <= end_dt]
        
        # Get latest record for each user
        latest_credit = credit_df.sort_values("timestamp").groupby("user_id").last().reset_index()
        latest_tx = tx_df.sort_values("timestamp").groupby("user_id").last().reset_index()
        
        # Merge latest features
        merged_latest = pd.merge(latest_credit, latest_tx, on="user_id", suffixes=("_credit", "_tx"), how="outer")
        
        # Update SQLite database
        for _, row in merged_latest.iterrows():
            user_id = int(row["user_id"])
            credit = int(row["credit_score"]) if not pd.isna(row["credit_score"]) else None
            tx = float(row["transaction_volume_30d"]) if not pd.isna(row["transaction_volume_30d"]) else None
            
            # Upsert into SQLite
            self.cursor.execute("""
                INSERT INTO online_user_features (user_id, credit_score, transaction_volume_30d, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    credit_score = excluded.credit_score,
                    transaction_volume_30d = excluded.transaction_volume_30d,
                    last_updated = excluded.last_updated
            """, (user_id, credit, tx, end_dt.isoformat()))
            
        self.conn.commit()
        print("✅ Online Store materialized successfully.")

    def get_online_features(
        self, 
        entity_keys: List[Dict[str, int]], 
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Queries the online SQLite store to fetch low-latency features.
        """
        results = []
        features_str = ", ".join(feature_names)
        
        for entity in entity_keys:
            user_id = entity["user_id"]
            self.cursor.execute(
                f"SELECT user_id, {features_str} FROM online_user_features WHERE user_id = ?", 
                (user_id,)
            )
            row = self.cursor.fetchone()
            if row:
                res = {"user_id": row[0]}
                for i, name in enumerate(feature_names):
                    res[name] = row[i+1]
                results.append(res)
            else:
                # Fallback / Empty
                res = {"user_id": user_id}
                for name in feature_names:
                    res[name] = None
                results.append(res)
                
        return results

# %% [markdown]
# ## 📁 2. Simulating the S3 Offline Store Feature Data
# Let's generate historical timeseries feature values for credit scores and transaction volumes, and write them to our simulated S3 offline store.

# %%
# Define sample timeseries user feature history
credit_scores_history = pd.DataFrame([
    {"user_id": 1001, "timestamp": "2026-01-01 00:00:00", "credit_score": 650},
    {"user_id": 1001, "timestamp": "2026-03-01 00:00:00", "credit_score": 670},
    {"user_id": 1001, "timestamp": "2026-05-01 00:00:00", "credit_score": 710},
    
    {"user_id": 1002, "timestamp": "2026-01-01 00:00:00", "credit_score": 720},
    {"user_id": 1002, "timestamp": "2026-04-01 00:00:00", "credit_score": 700},
])

transaction_volume_history = pd.DataFrame([
    {"user_id": 1001, "timestamp": "2026-01-01 00:00:00", "transaction_volume_30d": 1200.50},
    {"user_id": 1001, "timestamp": "2026-02-01 00:00:00", "transaction_volume_30d": 1500.00},
    {"user_id": 1001, "timestamp": "2026-04-01 00:00:00", "transaction_volume_30d": 1800.75},
    
    {"user_id": 1002, "timestamp": "2026-01-01 00:00:00", "transaction_volume_30d": 5000.00},
    {"user_id": 1002, "timestamp": "2026-03-01 00:00:00", "transaction_volume_30d": 5200.00},
])

# %% [markdown]
# ## 🛡️ 3. Point-in-Time Correctness vs. Data Leakage
# Data leakage happens when a model is trained using features that weren't yet available at the time of prediction.
# For example, if User 1001 applies for a loan on **March 15, 2026**:
# - **Correct Credit Score:** 670 (updated on March 1).
# - **Data Leakage Credit Score:** 710 (updated on May 1). If we join naively without timestamps, we use the 710 credit score, which is a leak from the future!
#
# Let's run a comparison showing how a standard join fails, while an **ASOF Join** retrieves the correct point-in-time value.

# %%
# Entity Dataframe: Represents events (e.g. loan application requests)
loan_applications = pd.DataFrame([
    {"user_id": 1001, "timestamp": "2026-03-15 12:00:00", "action": "loan_request"},
    {"user_id": 1002, "timestamp": "2026-02-15 09:00:00", "action": "loan_request"},
])

print("📋 Entity events (Loan applications):")
print(loan_applications)

# %%
# Run the simulation inside mock S3 context
with mock_aws():
    # Setup mock S3 environment
    bucket = "mlops-feature-store"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)
    
    # Initialize Feature Store
    fs = SimulatedFeatureStore(s3_bucket=bucket)
    
    # Populate S3 Offline Store
    fs.upload_offline_features("credit_scores", credit_scores_history)
    fs.upload_offline_features("transaction_volumes", transaction_volume_history)
    
    # 1. Retrieve Historical Features (ASOF Join)
    historical_df = fs.get_historical_features(
        entity_df=loan_applications,
        feature_names=["credit_scores", "transaction_volumes"]
    )
    
    print("\n🛡️ Point-in-Time Correct Joined Features (Offline Store):")
    print(historical_df)
    
    # Verification assertions
    user_1001_row = historical_df[historical_df["user_id"] == 1001].iloc[0]
    user_1002_row = historical_df[historical_df["user_id"] == 1002].iloc[0]
    
    # For User 1001 applying on March 15:
    # Latest credit score before March 15 is 670 (from March 1). NOT 710 (from May 1).
    # Latest transaction volume before March 15 is 1500.00 (from Feb 1). NOT 1800.75 (from April 1).
    assert user_1001_row["credit_score"] == 670, f"Leakage! Expected credit score 670, got {user_1001_row['credit_score']}"
    assert user_1001_row["transaction_volume_30d"] == 1500.00
    
    # For User 1002 applying on Feb 15:
    # Latest credit score before Feb 15 is 720 (from Jan 1).
    # Latest transaction volume before Feb 15 is 5000.00 (from Jan 1). NOT 5200.00 (from March 1).
    assert user_1002_row["credit_score"] == 720
    assert user_1002_row["transaction_volume_30d"] == 5000.00
    print("\n✅ Verification Successful: Zero data leakage occurred during training set generation!")

# %% [markdown]
# ## 🔄 4. Materializing to the Online Store
# When serving models in production, running complex S3 parquet file downloads and as-of joins is too slow. 
# We need to *materialize* the latest feature values up to the current timestamp into a low-latency database (SQLite online store).

# %%
with mock_aws():
    # Setup mock S3
    bucket = "mlops-feature-store"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)
    
    fs = SimulatedFeatureStore(s3_bucket=bucket)
    fs.upload_offline_features("credit_scores", credit_scores_history)
    fs.upload_offline_features("transaction_volumes", transaction_volume_history)
    
    # Materialize features up to April 15, 2026
    # This should load:
    # - User 1001: credit_score = 670 (since May 1 has not happened), tx_volume = 1800.75 (updated on April 1)
    # - User 1002: credit_score = 700 (updated on April 1), tx_volume = 5200.00 (updated on March 1)
    fs.materialize(end_timestamp="2026-04-15 00:00:00")
    
    # Let's inspect the Online SQLite Database content
    print("\n🔍 Querying SQLite Online Database directly:")
    db_df = pd.read_sql_query("SELECT * FROM online_user_features", fs.conn)
    print(db_df)
    
    # 2. Retrieve Online Features (O(1) lookup for serving API)
    online_features = fs.get_online_features(
        entity_keys=[{"user_id": 1001}, {"user_id": 1002}],
        feature_names=["credit_score", "transaction_volume_30d"]
    )
    
    print("\n⚡ Low-Latency Online Features retrieved for prediction:")
    print(online_features)
    
    # Assertions to ensure materialization worked correctly
    assert online_features[0]["credit_score"] == 670
    assert online_features[0]["transaction_volume_30d"] == 1800.75
    assert online_features[1]["credit_score"] == 700
    assert online_features[1]["transaction_volume_30d"] == 5200.00
    print("\n✅ Verification Successful: Online store holds correct and fresh features!")

# %% [markdown]
# ## 🖥️ 6. Unified CLI Feature Store Integration
# Module 13 extends our CLI with `mlops feature` to manage simulated Feature Store tasks:
# * **Apply feature schema configurations:**
#   ```bash
#   uv run mlops feature apply
#   ```
# * **Materialize features to SQLite online database:**
#   ```bash
#   uv run mlops feature materialize
#   ```
# * **Fetch real-time features for low-latency inference:**
#   ```bash
#   uv run mlops feature get
#   ```
#
# Let's verify the help information for feature store operations on our unified CLI:

# %%
import subprocess
result = subprocess.run(["mlops", "feature", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we've set up the Feature Store, let's step into the gRPC, Batch & Release Strategies guide to explore low-latency gRPC serving!

