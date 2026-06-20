# %% [markdown]
# # 📦 Data Versioning with DVC
#
# This tutorial demonstrates how to version control data files using Data Version Control (DVC).
#
# ### The DVC Concept: Separating Metadata from Weights
#
# Traditional version control systems like Git struggle with large datasets and binary weights. Committing large files causes Git databases to grow permanently, slowing down command performance and making cloning impractical.
#
# DVC solves this by storing the large files in an external object storage system, computing their MD5 hashes, and writing lightweight `.dvc` text-based pointers to the Git repository.
#
# ```mermaid
#  graph TD
#      subgraph git_repository_tracks_source_pointers ["Git Repository (Tracks Source & Pointers)"]
#          A["Git Tracking System"] -->|"Commit"| B["train.py"]
#          A -->|"Commit"| C["housing_raw.csv.dvc"]
#          C -->|"Text Metadata"| D["md5: 8a4c28b5..."]
#      end
#
#      subgraph local_workspace_excluded_from_git ["Local Workspace (Excluded from Git)"]
#          E["housing_raw.csv"] -.->|Auto-listed in .gitignore| A
#          E -->|"dvc add / hashing"| F["Local DVC Cache .dvc/cache/"]
#      end
#
#      subgraph remote_backend_s3_storage ["Remote Backend (S3 Storage)"]
#          F -->|"dvc push"| G["Production AWS S3 Bucket"]
#          G -->|"dvc pull"| F
#          F -->|"Restore"| E
#      end
#
# ```
#
# In this module, we will explore:
# 1. Initializing DVC inside the repository.
# 2. Tracking synthetic raw datasets in `data/raw/` and excluding them from git.
# 3. Configuring a local remote directory mimicking production cloud storage.


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

import pandas as pd
import numpy as np
import docker
from dvc.repo import Repo

# Ensure clean setup
os.makedirs("data", exist_ok=True)

# Generate synthetic dataset representing housing prices
np.random.seed(42)
num_samples = 100
housing_data = pd.DataFrame({
    "area_sqft": np.random.randint(800, 3500, size=num_samples),
    "bedrooms": np.random.randint(1, 6, size=num_samples),
    "price_usd": np.random.randint(150000, 800000, size=num_samples)
})

raw_data_path = "data/housing_raw.csv"
housing_data.to_csv(raw_data_path, index=False)
print(f"✅ Generated synthetic dataset at {raw_data_path} ({len(housing_data)} rows)")

# %% [markdown]
# ## ⚙️ 1. Initialize DVC
# We will check if DVC is already initialized. If not, we will initialize it.
# *(Note: If you run this inside a Git repository, DVC integrates with Git hooks automatically)*
#
# ```altshell
# dvc init --subdir
# ```

# %%
# Initialize DVC natively if .dvc directory does not exist
if not os.path.exists(".dvc"):
    print("🚀 Initializing DVC...")
    Repo.init(subdir=True)
else:
    print("✅ DVC is already initialized.")

# Instantiate DVC Repo manager
repo = Repo(".")

# %% [markdown]
# ## 📦 2. Versioning the Dataset
# Now we track our dataset `housing_raw.csv` using DVC. This will:
# 1. Create a `data/housing_raw.csv.dvc` file.
# 2. Add `data/housing_raw.csv` to `.gitignore` automatically.
#
# ```altshell
# dvc add data/housing_raw.csv
# ```

# %%
print("📦 Adding dataset to DVC tracking...")
repo.add(raw_data_path)

# Let's inspect the created pointer file
dvc_pointer_path = f"{raw_data_path}.dvc"
if os.path.exists(dvc_pointer_path):
    print(f"\n📄 Contents of {dvc_pointer_path}:")
    with open(dvc_pointer_path, "r") as f:
        print(f.read())

# %% [markdown]
# ## 🗄️ 3. Configuring a Simulated Remote Storage
# In production, we would use an AWS S3 bucket, Google Cloud Storage, or Azure Blob storage.
# For local tutorials, we can set up a local folder outside of git control (e.g., `local_remote`) as our mock remote storage.
#
# ```altshell
# dvc remote add -d myremote local_remote --force
# ```

# %%
remote_dir = "local_remote"
os.makedirs(remote_dir, exist_ok=True)

print(f"⚙️ Configuring DVC local remote at '{remote_dir}'...")
with repo.config.edit() as conf:
    conf.setdefault("core", {})["remote"] = "myremote"
    conf.setdefault("core", {})["autostage"] = True
    conf.setdefault("remote", {}).setdefault("myremote", {})["url"] = remote_dir

# %% [markdown]
# ## 🔄 4. Simulating DVC Push and Pull
# Now let's push the tracked data to our simulated remote.
#
# ```altshell
# dvc push
# ```

# %%
print("⬆️ Pushing data to DVC remote...")
repo.push()

# %% [markdown]
# Let's prove DVC is versioning the data. If we delete the raw CSV and run `dvc pull`, DVC will restore it!
#
# ```altshell
# rm data/housing_raw.csv
# dvc pull
# ```

# %%
print("🧹 Deleting the raw dataset file...")
if os.path.exists(raw_data_path):
    os.remove(raw_data_path)
    print(f"Deleted {raw_data_path}. File exists: {os.path.exists(raw_data_path)}")

print("\n⬇️ Restoring the dataset using DVC pull...")
repo.pull()

# Confirm restoration
if os.path.exists(raw_data_path):
    print(f"✅ Successfully restored raw data file. Rows: {len(pd.read_csv(raw_data_path))}")
else:
    print("❌ Failed to restore data.")


# %% [markdown]
# ## 🌐 5. Advanced: Dockerized Mock S3 & CLI-based Data Extraction
# In production enterprise environments, data might be ingested into an S3 bucket or similar cloud object store.
# To simulate this architecture, we will:
# 1. Set up a local Docker network (`mlops-net`).
# 2. Build and run a Mock S3 server container using `moto`.
# 3. Build a Dockerized importer/tracker CLI container that uploads a raw text corpus (`data/raw/FM_3-05-2025.md`) to the mock S3 bucket.
# 4. Extract/download the corpus from mock S3 to a local output path (`data/extracted/FM_3-05-2025.md`).
# 5. Run the importer/tracker container to automatically version-control and track the extracted file using DVC inside the mounted workspace directory.

# %%
import time

# We must locate the module directory to build docker containers correctly, even when running via mkdocs / jupyter
base_dir = os.getcwd()
module_dir = None
for p in ["src/data_experimentation/module_04_data_versioning_dvc", "."]:
    if os.path.exists(os.path.join(base_dir, p, "Dockerfile.mock_s3")):
        module_dir = os.path.abspath(os.path.join(base_dir, p))
        break

if not module_dir:
    module_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

print(f"Working Directory: {base_dir}")
print(f"Module Directory: {module_dir}")

# Initialize native Docker client
docker_client = docker.from_env()

# %% [markdown]
# ### Step 5.1: Create local Docker network
# We create a dedicated Docker network to allow the S3 server and CLI container to communicate.
#
# ```altshell
# docker network create mlops-net
# ```

# %%
try:
    print("🌐 Creating Docker network 'mlops-net'...")
    docker_client.networks.create("mlops-net", driver="bridge")
    print("✅ Network 'mlops-net' is ready.")
except Exception as e:
    # If network already exists, retrieve it
    try:
        docker_client.networks.get("mlops-net")
        print("✅ Network 'mlops-net' already exists and is ready.")
    except Exception:
        print(f"Network status: {e}")

# %% [markdown]
# ### Step 5.2: Build and run the Mock S3 Docker container
# We compile the lightweight `Dockerfile.mock_s3` and launch it.
#
# ```altshell
# docker build -t mock-s3-server -f Dockerfile.mock_s3 .
# docker run -d --name mock-s3 --network mlops-net -p 5000:5000 mock-s3-server
# ```

# %%
# Build only if the image doesn't exist to avoid redundant builds and dangling images
try:
    docker_client.images.get("mock-s3-server")
    print("✅ Docker image 'mock-s3-server' already exists. Skipping build.")
except docker.errors.ImageNotFound:
    print("🏗️ Building Mock S3 server image...")
    docker_client.images.build(
        path=module_dir,
        dockerfile="Dockerfile.mock_s3",
        tag="mock-s3-server"
    )

# Run only if mock-s3 container doesn't exist or is not running
try:
    s3_container = docker_client.containers.get("mock-s3")
    if s3_container.status != "running":
        print("🚀 Starting existing Mock S3 server container...")
        s3_container.start()
    else:
        print("✅ Mock S3 server container is already running.")
except docker.errors.NotFound:
    print("🚀 Running Mock S3 server container...")
    s3_container = docker_client.containers.run(
        "mock-s3-server",
        name="mock-s3",
        network="mlops-net",
        ports={"5000/tcp": 5000},
        detach=True
    )

# Give mock-s3 container a couple of seconds to initialize and boot up
print("⏳ Waiting for S3 server to boot...")
time.sleep(3)

# %% [markdown]
# ### Step 5.3: Build the Dockerized Data Importer and Tracker container
# Next, we build the client container that wraps our CLI script (`import_tracker.py`).
#
# ```altshell
# docker build -t data-importer -f Dockerfile.importer .
# ```

# %%
# Build only if the image doesn't exist to avoid redundant builds and dangling images
try:
    docker_client.images.get("data-importer")
    print("✅ Docker image 'data-importer' already exists. Skipping build.")
except docker.errors.ImageNotFound:
    print("🏗️ Building Importer & Tracker client image...")
    docker_client.images.build(
        path=module_dir,
        dockerfile="Dockerfile.importer",
        tag="data-importer"
    )

# %% [markdown]
# ### Step 5.4: Import/Upload the corpus file to Mock S3
# We mount our data folder, and run the container to upload our corpus file `data/raw/FM_3-05-2025.md` to `s3://corpus-bucket/FM_3-05-2025.md`.
#
# ```altshell
# docker run --rm --network mlops-net \
#   -e AWS_S3_ENDPOINT_URL=http://mock-s3:5000 \
#   -v $(pwd)/data:/app/data \
#   data-importer upload /app/data/raw/FM_3-05-2025.md s3://corpus-bucket/FM_3-05-2025.md
# ```

# %%
corpus_s3_url = "s3://corpus-bucket/FM_3-05-2025.md"
corpus_raw_path = "data/raw/FM_3-05-2025.md"

print(f"🚀 Uploading local {corpus_raw_path} to mock S3...")
upload_logs = docker_client.containers.run(
    "data-importer",
    command=f"upload /app/{corpus_raw_path} {corpus_s3_url}",
    network="mlops-net",
    environment={"AWS_S3_ENDPOINT_URL": "http://mock-s3:5000"},
    volumes={f"{base_dir}/data": {"bind": "/app/data", "mode": "rw"}},
    remove=True
)
print(upload_logs.decode().strip())

# %% [markdown]
# ### Step 5.5: Extract/Download the corpus from Mock S3
# Now we simulate pulling data from external storage into our extraction zone `data/extracted/FM_3-05-2025.md`.
#
# ```altshell
# docker run --rm --network mlops-net \
#   -e AWS_S3_ENDPOINT_URL=http://mock-s3:5000 \
#   -v $(pwd)/data:/app/data \
#   data-importer extract s3://corpus-bucket/FM_3-05-2025.md /app/data/extracted/FM_3-05-2025.md
# ```

# %%
corpus_extracted_path = "data/extracted/FM_3-05-2025.md"

print(f"📥 Extracting {corpus_s3_url} to local {corpus_extracted_path}...")
extract_logs = docker_client.containers.run(
    "data-importer",
    command=f"extract {corpus_s3_url} /app/{corpus_extracted_path}",
    network="mlops-net",
    environment={"AWS_S3_ENDPOINT_URL": "http://mock-s3:5000"},
    volumes={f"{base_dir}/data": {"bind": "/app/data", "mode": "rw"}},
    remove=True
)
print(extract_logs.decode().strip())

# Confirm file extraction
local_extracted_full_path = os.path.join(base_dir, corpus_extracted_path)
if os.path.exists(local_extracted_full_path):
    print(f"🎉 Success! Extracted file size: {os.path.getsize(local_extracted_full_path)} bytes")
else:
    print("❌ Error: Extracted file was not found!")

# %% [markdown]
# ### Step 5.6: Track the Extracted File with DVC in Docker
# Finally, we track the extracted file under version control.
# We mount the full project directory and run the `track` subcommand.
#
# ```altshell
# docker run --rm -v $(pwd):/workspace -w /workspace data-importer track data/extracted/FM_3-05-2025.md
# ```

# %%
print(f"📦 Version tracking {corpus_extracted_path} with DVC inside Docker...")
track_logs = docker_client.containers.run(
    "data-importer",
    command=f"track {corpus_extracted_path}",
    volumes={f"{base_dir}": {"bind": "/workspace", "mode": "rw"}},
    working_dir="/workspace",
    remove=True
)
print(track_logs.decode().strip())

# Inspect the resulting .dvc file
dvc_meta_path = f"{local_extracted_full_path}.dvc"
if os.path.exists(dvc_meta_path):
    print(f"\n📄 Contents of {corpus_extracted_path}.dvc:")
    with open(dvc_meta_path, "r") as f:
        print(f.read())
else:
    print("❌ Error: DVC pointer file not created!")

# %% [markdown]
# ### 🖥️ 6. Unified CLI Integration for DVC
# Module 4 extends our CLI by adding DVC operations via `mlops dvc`:
# * **Initialize DVC:**
#   ```bash
#   uv run mlops dvc init
#   ```
# * **Track a file:**
#   ```bash
#   uv run mlops dvc track data/housing_raw.csv
#   ```
# * **Push to remote / Pull from remote:**
#   ```bash
#   uv run mlops dvc push
#   uv run mlops dvc pull
#   ```
# * **Run DVC commands via Docker container:**
#   ```bash
#   docker run --rm -v $(pwd):/workspace -w /workspace mlops-cli dvc pull
#   ```
#
# Let's verify the help information for DVC operations on our unified CLI:

# %%
import subprocess
result = subprocess.run(["mlops", "dvc", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that you've mastered data versioning and the CLI integration, let's step into the Experiment Tracking and Model Registry guides with MLflow!

# %% [markdown]
# ### Step 5.7: Tear Down Container Resources (Optional)
# To keep the workspace clean, you can stop and remove the S3 container and the Docker network when you are done.
#
# ```altshell
# docker rm -f mock-s3
# docker network rm mlops-net
# ```

# %%
# Commented out by default to allow exploring the running containers and avoid rebuilding them on every page compilation.
# print("🧹 Tearing down mock docker resources...")
# try:
#     s3_container = docker_client.containers.get("mock-s3")
#     s3_container.remove(force=True)
# except docker.errors.NotFound:
#     pass
# except Exception as e:
#     print(f"Error removing container: {e}")
# 
# try:
#     net = docker_client.networks.get("mlops-net")
#     net.remove()
# except docker.errors.NotFound:
#     pass
# except Exception as e:
#     print(f"Error removing network: {e}")
# 
# print("✨ Sandbox cleanup finished successfully!")
