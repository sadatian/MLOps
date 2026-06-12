# %% [markdown]
# # AWS S3 Simulation with `boto3` & `moto`
#
# In production MLOps pipelines, AWS S3 is frequently used to store model weights, training logs, and feature store dumps.
# To test our MLOps code locally without paying for AWS or configuring credentials, we use **Moto** — a library that mocks AWS services.
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
# ## 1. Executing Simulated S3 Operations
# We will use the `mock_aws` context manager from Moto. Any boto3 client initialized inside this context is automatically mocked, and actions (like creating buckets or writing objects) take place in an in-memory virtual environment.

# %%
# Define test variables
bucket_name = "mlops-tutorial-artifacts"
object_key = "models/model_metadata.json"
sample_data = '{"model_name": "linear_regression", "accuracy": 0.89, "version": "v1"}'

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
    
    # 3. Upload object
    print(f"Uploading file metadata to 's3://{bucket_name}/{object_key}'...")
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=sample_data
    )
    
    # 4. Download and verify object
    print(f"Retrieving file from simulated bucket...")
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    retrieved_content = response["Body"].read().decode("utf-8")
    
    print("\n✅ Successfully retrieved object from Mock S3!")
    print(f"Retrieved Content: {retrieved_content}")

# %% [markdown]
# ## 2. Standalone Mock S3 Server (Optional)
# If you want a mock S3 endpoint that external processes (like DVC or CLI tools) can talk to via HTTP:
#
# You can run `moto_server` in your WSL terminal:
# ```bash
# # Run local mock S3 server on port 5000
# uv run moto_server s3 -p 5000
# ```
#
# And then configure DVC, MLflow, or boto3 to communicate with that endpoint:
# ```python
# s3_client = boto3.client(
#     "s3", 
#     endpoint_url="http://localhost:5000",
#     region_name="us-east-1"
# )
# ```
#
# Now that we know how to mock AWS locally, let's step into the Data Version Control (DVC) and Experiment Tracking guides to learn about data versioning and model registration!
