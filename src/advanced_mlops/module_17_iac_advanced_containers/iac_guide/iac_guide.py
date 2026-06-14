# %% [markdown]
# # 🛠️ Infrastructure as Code (IaC) & Advanced Containerization
#
# Deploying machine learning models to production requires reproducible, automated environments.
#
# ### The Unified MLOps Lifecycle Integration
#
# A production MLOps system requires close integration across three distinct core layers:
# 1. **Infrastructure as Code (IaC):** Declaratively defining backend resources (S3 buckets, metadata databases, orchestration servers, GPU-virtual networks) in code files using **Terraform**.
# 2. **Data & Pipeline Versioning:** Storing data profiles and pipelines using **DVC** inside the S3 storage bucket provisioned by Terraform.
# 3. **Experimentation & Model Registry:** Tracking metric logs and hosting candidate model packages inside the database and S3 buckets provisioned by Terraform.
#
# ```mermaid
#  graph TD
#      subgraph infrastructure_provisioning_terraform_iac ["Infrastructure Provisioning (Terraform IaC)"]
#          A["Terraform Configuration: main.tf"] -->|"terraform apply"| B{"AWS Sim API: LocalStack / Moto"}
#          B -->|"Provisions DVC backend"| C["S3 Bucket: mlops-dvc-remote"]
#          B -->|"Provisions MLflow backend"| D["S3 Bucket: mlflow-artifact-store"]
#          B -->|"Provisions registry metadata"| E["DynamoDB Table: model-metadata"]
#      end
#
#      subgraph data_versioning_pipelines_dvc ["Data Versioning & Pipelines (DVC)"]
#          F["dvc.yaml Pipeline Execution"] -->|"Inputs/Outputs"| G["Local cache .dvc/cache"]
#          G -->|"dvc push / data backup"| C
#      end
#
#      subgraph experiment_tracking_registry_mlflow ["Experiment Tracking & Registry (MLflow)"]
#          H["Training Run / mlflow.log_model"] -->|"Upload weights"| D
#          H -->|"Log metrics & parameters"| E
#          I["Model Registry Staging/Prod promotion"] -->|"Update registration"| E
#      end
#
#      subgraph containerized_inference_serving_layer ["Containerized Inference (Serving Layer)"]
#          J["FastAPI Serving Container"] -->|"1. Query production model URI"| I
#          J -->|"2. Download weight model.pkl"| D
#          J -->|"3. Route inferences"| K["Downstream Consumers / Users"]
#      end
#
#      style C fill:#bbdefb,stroke:#1976d2,stroke-width:1.5px
#      style D fill:#bbdefb,stroke:#1976d2,stroke-width:1.5px
#      style E fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#      style J fill:#d4edda,stroke:#28a745,stroke-width:2px
#
# ```
#
# In this module, we will implement:
# 1. Real-world **Terraform** configurations for a model registry and metadata serving system.
# 2. A programmatic **Terraform Validation Wrapper** that runs `terraform init` and `terraform validate`.
# 3. A **Policy as Code Validator** in Python to check Terraform compliance.
# 4. A **LocalStack/Moto Infrastructure Simulator** that reads declarative variables and spins them up.
# 5. A **GPU Dockerfile Validator** that ensures container files are optimized and GPU-ready.
# 6. A **CUDA Verification Check** with automatic CPU fallbacks.


# %%
import os
import re
import subprocess
import shutil
import sys
from typing import Dict, Any, List, Tuple
import boto3
from moto import mock_aws

# Paths for module files (support running as script or in interactive notebook)
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.path.abspath(os.getcwd())
    possible_paths = [
        current_dir,
        os.path.join(current_dir, "src", "advanced_mlops", "module_17_iac_advanced_containers"),
        os.path.join(current_dir, "docs", "src", "advanced_mlops", "module_17_iac_advanced_containers"),
        os.path.join(current_dir, "module_17_iac_advanced_containers"),
    ]
    BASE_DIR = current_dir
    for p in possible_paths:
        if os.path.isdir(os.path.join(p, "terraform")):
            BASE_DIR = p
            break

TF_DIR = os.path.join(BASE_DIR, "terraform")
DOCKERFILE_PATH = os.path.join(BASE_DIR, "Dockerfile.gpu")
MAIN_TF_PATH = os.path.join(TF_DIR, "main.tf")
VARIABLES_TF_PATH = os.path.join(TF_DIR, "variables.tf")

print(f"📂 Module directory: {BASE_DIR}")
print(f"🛠️ Terraform directory: {TF_DIR}")

# %% [markdown]
# ## 🛠️ 1. Terraform Execution & Validation
#
# We invoke the system's `terraform` binary to initialize our modules and validate syntax. 
# Running these checks ensures our declarative files are formatted properly and free of syntax errors before deployment.

# %%
def run_terraform_validation(tf_dir: str) -> Dict[str, Any]:
    """
    Runs terraform init and terraform validate on the given directory.
    Handles missing binary, lock issues, or network offline issues gracefully.
    """
    if not shutil.which("terraform"):
        return {
            "success": False,
            "error": "Terraform binary not found in system path."
        }
    
    # 1. Run terraform init
    try:
        init_res = subprocess.run(
            ["terraform", "init", "-backend=false"],
            cwd=tf_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        if init_res.returncode != 0:
            return {
                "success": False,
                "stage": "init",
                "stdout": init_res.stdout,
                "stderr": init_res.stderr,
                "error": "terraform init failed"
            }
    except Exception as e:
        return {
            "success": False,
            "stage": "init",
            "error": f"Exception during terraform init: {str(e)}"
        }

    # 2. Run terraform validate
    try:
        val_res = subprocess.run(
            ["terraform", "validate"],
            cwd=tf_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        return {
            "success": val_res.returncode == 0,
            "stage": "validate",
            "stdout": val_res.stdout,
            "stderr": val_res.stderr,
            "error": None if val_res.returncode == 0 else "terraform validate failed"
        }
    except Exception as e:
        return {
            "success": False,
            "stage": "validate",
            "error": f"Exception during terraform validate: {str(e)}"
        }

# Execute Terraform validation checks
print("⚙️ Running Terraform Syntax Validation...")
tf_val_results = run_terraform_validation(TF_DIR)
print(f"  Terraform Success: {tf_val_results.get('success')}")
if not tf_val_results.get("success"):
    print(f"  Error Detail: {tf_val_results.get('error') or tf_val_results.get('stderr')}")

# %% [markdown]
# ## 🛡️ 2. Policy as Code (Static Compliance Checks)
#
# In addition to basic syntax, enterprise MLOps architectures require policy compliance checks (e.g., security controls, tagging structures, provider locks).
# We implement a Python policy-as-code validator that scans our Terraform files.

# %%
def validate_terraform_policies(tf_file_path: str) -> Dict[str, Any]:
    """
    Static policy-as-code checker (mocking a tool like tfsec or terrascan).
    Reads the main.tf file and checks for security/standards compliance:
    1. S3 bucket encryption is declared.
    2. S3 bucket versioning is declared.
    3. DynamoDB table specifies PAY_PER_REQUEST billing mode.
    4. AWS provider endpoints redirect to LocalStack (localhost/127.0.0.1).
    """
    if not os.path.exists(tf_file_path):
        return {"success": False, "error": f"File {tf_file_path} not found"}
        
    with open(tf_file_path, "r") as f:
        content = f.read()

    checks = {}
    
    # 1. AWS Provider redirect check
    provider_has_localstack = False
    provider_blocks = re.findall(r'provider\s+"aws"\s*\{(.*?)\}', content, re.DOTALL)
    for block in provider_blocks:
        if "endpoints" in block and ("localhost" in block or "127.0.0.1" in block):
            provider_has_localstack = True
            break
    checks["provider_localstack_redirection"] = {
        "passed": provider_has_localstack,
        "details": "AWS provider endpoints must target LocalStack to prevent production data leaks."
    }

    # 2. S3 Bucket encryption check
    has_s3_encryption = "aws_s3_bucket_server_side_encryption_configuration" in content
    checks["s3_bucket_encryption"] = {
        "passed": has_s3_encryption,
        "details": "Model registry S3 bucket must have server-side encryption enabled for compliance."
    }

    # 3. S3 Bucket versioning check
    has_s3_versioning = "aws_s3_bucket_versioning" in content
    checks["s3_bucket_versioning"] = {
        "passed": has_s3_versioning,
        "details": "Model registry S3 bucket must have versioning enabled for easy deployment rollbacks."
    }

    # 4. DynamoDB Billing Mode
    has_pay_per_request = "billing_mode" in content and "PAY_PER_REQUEST" in content
    checks["dynamodb_cost_optimization"] = {
        "passed": has_pay_per_request,
        "details": "DynamoDB metadata table should use PAY_PER_REQUEST billing to minimize idle charges."
    }

    all_passed = all(check["passed"] for check in checks.values())
    return {
        "success": all_passed,
        "checks": checks
    }

# Execute policy-as-code compliance checks
print("🛡️ Running Static Policy Gating...")
policy_results = validate_terraform_policies(MAIN_TF_PATH)
print(f"  Overall Policy Passed: {policy_results['success']}")
for check_name, check_data in policy_results.get("checks", {}).items():
    status_emoji = "✅" if check_data["passed"] else "❌"
    print(f"  {status_emoji} {check_name}: {check_data['details']}")

# %% [markdown]
# ## ☁️ 3. LocalStack Simulator (Moto & Boto3)
#
# When running test suites, we want to simulate provisioning declared resources locally.
# We parse the Terraform default variables, initialize our local simulator with `moto`, and simulate model registry transactions.

# %%
def parse_variables_file(var_file_path: str) -> Dict[str, str]:
    """
    Parses variable defaults from variables.tf to use in our simulation.
    """
    defaults = {}
    if not os.path.exists(var_file_path):
        return defaults
        
    with open(var_file_path, "r") as f:
        content = f.read()
        
    var_blocks = re.findall(r'variable\s+"([^"]+)"\s*\{(.*?)\}', content, re.DOTALL)
    for name, block in var_blocks:
        default_match = re.search(r'default\s*=\s*"([^"]+)"', block)
        if default_match:
            defaults[name] = default_match.group(1)
            
    return defaults


class LocalStackSimulator:
    """
    Simulates a LocalStack environment using Moto.
    It reads Terraform defaults, provisions resources via boto3,
    and runs client tests against the simulated cloud services.
    """
    def __init__(self, bucket_name: str, table_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.table_name = table_name
        self.region = region
        self.s3_client = None
        self.dynamodb_client = None

    def provision_resources(self):
        """
        Simulates Terraform apply by creating the S3 bucket and DynamoDB table.
        """
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.dynamodb_client = boto3.client("dynamodb", region_name=self.region)
        
        # Create S3 Bucket
        if self.region == "us-east-1":
            self.s3_client.create_bucket(Bucket=self.bucket_name)
        else:
            self.s3_client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.region}
            )
            
        # Create DynamoDB Table
        self.dynamodb_client.create_table(
            TableName=self.table_name,
            KeySchema=[
                {"AttributeName": "ModelId", "KeyType": "HASH"},
                {"AttributeName": "Version", "KeyType": "RANGE"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "ModelId", "AttributeType": "S"},
                {"AttributeName": "Version", "AttributeType": "S"}
            ],
            BillingMode="PAY_PER_REQUEST"
        )
        print(f"🚀 [SIMULATION] Created S3 bucket: '{self.bucket_name}'")
        print(f"🚀 [SIMULATION] Created DynamoDB table: '{self.table_name}'")

    def upload_model_artifact(self, model_id: str, version: str, artifact_path: str, model_data: bytes):
        """
        Simulates model server uploading model checkpoints and metadata.
        """
        s3_key = f"models/{model_id}/{version}/{artifact_path}"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=model_data
        )
        
        s3_uri = f"s3://{self.bucket_name}/{s3_key}"
        self.dynamodb_client.put_item(
            TableName=self.table_name,
            Item={
                "ModelId": {"S": model_id},
                "Version": {"S": version},
                "S3Uri": {"S": s3_uri},
                "Status": {"S": "Registered"}
            }
        )
        print(f"📦 [REGISTRY] Uploaded artifact to {s3_uri} and logged metadata.")

    def fetch_model_metadata(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        Fetches the metadata record from DynamoDB.
        """
        res = self.dynamodb_client.get_item(
            TableName=self.table_name,
            Key={
                "ModelId": {"S": model_id},
                "Version": {"S": version}
            }
        )
        return res.get("Item", {})

# Run simulation test within a mock_aws context
print("\n☁️ Running simulated AWS environment operations...")
with mock_aws():
    tf_vars = parse_variables_file(VARIABLES_TF_PATH)
    bucket = tf_vars.get("bucket_name", "fallback-bucket")
    table = tf_vars.get("table_name", "fallback-table")
    region = tf_vars.get("aws_region", "us-east-1")
    
    sim = LocalStackSimulator(bucket, table, region)
    sim.provision_resources()
    sim.upload_model_artifact(
        model_id="ResNet50_Classifier",
        version="v1.0",
        artifact_path="resnet50.onnx",
        model_data=b"dummy_weights_data_resnet"
    )
    metadata = sim.fetch_model_metadata("ResNet50_Classifier", "v1.0")
    print(f"🔍 [METADATA READ] S3 Location: {metadata.get('S3Uri', {}).get('S')}")

# %% [markdown]
# ## 🐳 4. GPU-Ready Dockerfile Policy Validation
#
# Containerizing model servers for GPUs requires specialized base layers (e.g. `nvidia/cuda`) and environment configurations.
# We validate our GPU Dockerfile using python to check compliance with hardware standards.

# %%
def validate_gpu_dockerfile(dockerfile_path: str) -> Dict[str, Any]:
    """
    Inspects a Dockerfile to ensure it meets specialized GPU container standards:
    1. Inherits from nvidia/cuda base image (or similar GPU base layer).
    2. Maps correct NVIDIA Container Toolkit variables (NVIDIA_VISIBLE_DEVICES).
    3. Cleans up apt caches (rm -rf /var/lib/apt/lists/*) to prevent oversized layers.
    """
    if not os.path.exists(dockerfile_path):
        return {"success": False, "error": f"Dockerfile {dockerfile_path} not found"}
        
    with open(dockerfile_path, "r") as f:
        lines = f.readlines()
        
    checks = {
        "gpu_base_image": {"passed": False, "details": "Dockerfile must use an 'nvidia/cuda' or GPU-ready base image."},
        "nvidia_variables": {"passed": False, "details": "Dockerfile must expose NVIDIA_VISIBLE_DEVICES for GPU scheduling."},
        "cache_cleanup": {"passed": False, "details": "Dockerfile should clear apt-get caches to reduce image sizes."}
    }
    
    for line in lines:
        cleaned = line.strip()
        # 1. Base image check
        if cleaned.startswith("FROM"):
            if "nvidia/cuda" in cleaned.lower() or "pytorch" in cleaned.lower() or "tensorflow" in cleaned.lower():
                checks["gpu_base_image"]["passed"] = True
        
        # 2. NVIDIA Toolkit variables check
        if "NVIDIA_VISIBLE_DEVICES" in cleaned:
            checks["nvidia_variables"]["passed"] = True
            
        # 3. Clean cache check
        if "rm -rf /var/lib/apt/lists/*" in cleaned:
            checks["cache_cleanup"]["passed"] = True
            
    all_passed = all(check["passed"] for check in checks.values())
    return {
        "success": all_passed,
        "checks": checks
    }

# Execute Dockerfile policy validations
print("\n🐳 Checking Dockerfile.gpu configuration...")
docker_results = validate_gpu_dockerfile(DOCKERFILE_PATH)
print(f"  Dockerfile Policy Passed: {docker_results['success']}")
for check_name, check_data in docker_results.get("checks", {}).items():
    status_emoji = "✅" if check_data["passed"] else "❌"
    print(f"  {status_emoji} {check_name}: {check_data['details']}")

# %% [markdown]
# ## 🖥️ 5. GPU Hardware Detection & CPU Fallback
#
# Our model serving script must inspect resources on boot, detect GPU features, and gracefully fall back to CPU modes if GPUs are unallocated or unavailable.
# We query hardware resources directly using:
# 1. PyTorch CUDA check (`torch.cuda.is_available()`)
# 2. System command scan (`nvidia-smi`)
# 3. Kernel module checks (`/proc/driver/nvidia/version`)

# %%
def verify_gpu_availability() -> Dict[str, Any]:
    """
    Inspects local hardware resources at boot time to detect NVIDIA GPUs.
    Verifies if CUDA/GPUs are available, retrieves device names, and falls back to CPU serving safely.
    """
    cuda_available = False
    device_count = 0
    device_name = "None"
    driver_version = "None"

    # 1. First attempt: check PyTorch CUDA support
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            driver_version = f"CUDA {torch.version.cuda}"
    except ImportError:
        pass

    # 2. Second attempt: Check via nvidia-smi if PyTorch is absent or CPU-only
    if not cuda_available:
        smi_path = shutil.which("nvidia-smi")
        if smi_path:
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if res.returncode == 0 and res.stdout.strip():
                    cuda_available = True
                    device_count = len(res.stdout.strip().split("\n"))
                    device_name = res.stdout.strip().split("\n")[0]
                    
                    # Also try to get the driver version
                    ver_res = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5
                    )
                    if ver_res.returncode == 0:
                        driver_version = ver_res.stdout.strip()
            except Exception:
                pass

    # 3. Third attempt: Check proc driver version file (Linux only)
    if not cuda_available and os.path.exists("/proc/driver/nvidia/version"):
        try:
            with open("/proc/driver/nvidia/version", "r") as f:
                content = f.read()
                version_match = re.search(r"Kernel Module\s+([\d\.]+)", content)
                if version_match:
                    cuda_available = True
                    device_count = 1
                    device_name = "NVIDIA GPU (Detected via proc)"
                    driver_version = version_match.group(1)
        except Exception:
            pass

    status = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "active_device": device_name,
        "driver_version": driver_version,
        "serving_mode": "GPU-accelerated Inference" if cuda_available else "CPU Fallback Mode (Slow)"
    }
    
    print(f"\n🖥️ [HARDWARE DETECTOR] System GPU Scan:")
    print(f"  CUDA/NVIDIA GPU Available: {status['cuda_available']}")
    print(f"  Device Count:             {status['device_count']}")
    print(f"  Active Device:            {status['active_device']}")
    print(f"  Driver/CUDA Version:      {status['driver_version']}")
    print(f"  Serving Mode:             {status['serving_mode']}")
    
    return status

# Run real hardware verification check on the system
verify_gpu_availability()

# %% [markdown]
# ---
#
# 🎉 **Module 17 Completed!** You have successfully implemented Infrastructure as Code checks with Terraform, simulated AWS services via Moto, validated GPU-specialized container policies, and scanned real local hardware resources for GPU execution.
