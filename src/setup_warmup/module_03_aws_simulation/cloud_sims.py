# %% [markdown]
# # ☁️ Cloud Services Simulations and Mock Servers
#
# In production MLOps pipelines, cloud services like AWS S3 are frequently used to store model weights, training logs, and feature store dumps.
# To test our MLOps code locally without paying for AWS or configuring credentials, we use **Moto** — a library that mocks AWS services.
#
# ### The AWS Simulation / Interception Loop
#
# Standard cloud integration test suites require real cloud resources, leading to test latency, resource cost, and setup complexity. 
# `moto` solves this by programmatically intercepting all HTTP calls made by the `boto3` library, redirecting them to an in-memory virtual state machine that mimics AWS behavior.
#
# ```mermaid
# stateDiagram-v2
#     [*] --> ScriptExecution : Start Python script
#
#     state "boto3 Client Call" as Boto3Call
#     ScriptExecution --> Boto3Call : s3.create_bucket()
#
#     state "Socket Interception Layer" as InterceptLayer
#     Boto3Call --> InterceptLayer : HTTP Request dispatched
#
#     state "Moto In-Memory Simulation" as MotoMock {
#         [*] --> LocalRouting : Diverted at socket level
#         LocalRouting --> VirtualStateUpdate : Update in-memory S3 state
#         VirtualStateUpdate --> MockResponse : Generate simulated response
#     }
#
#     state "AWS Production Cloud" as RealAWS {
#         [*] --> InternetRouting : Transmit over public network
#         InternetRouting --> AWSAPI : Reach regional API endpoint
#         AWSAPI --> DiskWrite : Write to physical storage
#         DiskWrite --> AWSResponse : Return AWS response
#     }
#
#     InterceptLayer --> MotoMock : mock_aws is active
#     InterceptLayer --> RealAWS : mock_aws is inactive
#
#     state "Parse HTTP Response" as ParseResponse
#     MotoMock --> ParseResponse : Mock response returned
#     RealAWS --> ParseResponse : Real response returned
#
#     ParseResponse --> ScriptExecution : Return dict/result to user code
#     ScriptExecution --> [*] : Script completes
# ```
#
#
# In this module, we will explore:
# 1. Setting up mock AWS environment variables.
# 2. Initializing a mock S3 client using `boto3`.
# 3. Wrapping client code using `moto.mock_aws`.
# 4. Creating a simulated S3 bucket, uploading a file, and downloading it.


# %%
import os
import boto3
from moto import mock_aws

# Set dummy environment variables to prevent boto3 from attempting to connect to real AWS
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# %% [markdown]
# ## ☁️ 1. Executing Simulated S3 Operations
# We will use the `mock_aws` context manager from Moto. Any boto3 client initialized inside this context is automatically mocked, and actions (like creating buckets or writing objects) take place in an in-memory virtual environment.

# %%
# Define test variables for model registry
bucket_name = "mlops-model-registry"
object_key = "models/housing_model.pkl"

# Simulate a trained model artifact (serialized as a pickle file)
import pickle
sample_model = {
    "model_name": "housing_price_predictor",
    "features": ["area_k_sqft", "bedrooms"],
    "weights": [120.5, 45000.0],
    "intercept": 50000.0
}
serialized_model = pickle.dumps(sample_model)

print("🚀 Starting mock AWS S3 session...")

with mock_aws():
    # 1. Initialize S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")
    
    # 2. Create bucket
    print(f"Creating mock S3 bucket: '{bucket_name}'...")
    s3_client.create_bucket(Bucket=bucket_name)
    
    # Verify bucket creation
    buckets = s3_client.list_buckets()["Buckets"]
    print(f"Active buckets in simulated AWS account: {[b['Name'] for b in buckets]}")
    
    # 3. Upload model artifact
    print(f"Uploading serialized model artifact to 's3://{bucket_name}/{object_key}'...")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=serialized_model
    )
    
    # 4. Download and verify model artifact (mimicking how model serving fetches it)
    print(f"Retrieving model artifact from simulated bucket...")
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    retrieved_bytes = response["Body"].read()
    retrieved_model = pickle.loads(retrieved_bytes)
    
    print("\n✅ Successfully retrieved model artifact from Mock S3!")
    print(f"Retrieved Model Structure: {retrieved_model}")

# %% [markdown]
# ## 🖥️ 2. Standalone Mock S3 Server with the CLI
# If you want a mock S3 endpoint that external processes (like DVC or CLI tools) can talk to via HTTP:
#
# Module 3 extends the CLI by introducing `mlops moto s3` which starts a standalone local AWS S3 endpoint.
#
# * **Start local mock S3 server on port 5001:**
#   ```bash
#   uv run mlops moto s3 -p 5001
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm -it -p 5001:5001 mlops-cli moto s3 -p 5001
#   ```
#
# And then configure DVC, MLflow, or boto3 to communicate with that endpoint:
# ```python
# s3_client = boto3.client(
#     "s3", 
#     endpoint_url="http://localhost:5001",
#     region_name="us-east-1"
# )
# ```
#
# Let's verify the help description for the moto command:

# %%
import subprocess
result = subprocess.run(["mlops", "moto", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that we know how to mock AWS locally using our CLI, let's step into the Data Version Control (DVC) and Experiment Tracking guides to learn about data versioning and model registration!
