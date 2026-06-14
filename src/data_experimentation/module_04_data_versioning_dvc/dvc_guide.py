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
import os
import subprocess
import pandas as pd
import numpy as np

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
# We will check if DVC is already initialized. If not, we will run `dvc init` via subprocess.
# *(Note: If you run this inside a Git repository, DVC integrates with Git hooks automatically)*

# %%
def run_command(cmd_list):
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=True)
        print(f"Command Output:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Command Error Output:\n{e.stderr.strip()}")
        raise e

# Initialize DVC if .dvc directory does not exist
if not os.path.exists(".dvc"):
    print("🚀 Initializing DVC...")
    # --no-scm lets us initialize dvc even if git is not set up, but git is available, so we run standard init
    run_command(["dvc", "init", "--subdir"])
else:
    print("✅ DVC is already initialized.")

# %% [markdown]
# ## 📦 2. Versioning the Dataset
# Now we track our dataset `housing_raw.csv` using DVC. This will:
# 1. Create a `data/housing_raw.csv.dvc` file.
# 2. Add `data/housing_raw.csv` to `.gitignore` automatically.

# %%
print("📦 Adding dataset to DVC tracking...")
run_command(["dvc", "add", raw_data_path])

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

# %%
remote_dir = "local_remote"
os.makedirs(remote_dir, exist_ok=True)

print(f"⚙️ Configuring DVC local remote at '{remote_dir}'...")
# Add remote named 'myremote'
run_command(["dvc", "remote", "add", "-d", "myremote", remote_dir, "--force"])

# %% [markdown]
# ## 🔄 4. Simulating DVC Push and Pull
# Now let's push the tracked data to our simulated remote.

# %%
print("⬆️ Pushing data to DVC remote...")
run_command(["dvc", "push"])

# %% [markdown]
# Let's prove DVC is versioning the data. If we delete the raw CSV and run `dvc pull`, DVC will restore it!

# %%
print("🧹 Deleting the raw dataset file...")
if os.path.exists(raw_data_path):
    os.remove(raw_data_path)
    print(f"Deleted {raw_data_path}. File exists: {os.path.exists(raw_data_path)}")

print("\n⬇️ Restoring the dataset using DVC pull...")
run_command(["dvc", "pull"])

# Confirm restoration
if os.path.exists(raw_data_path):
    print(f"✅ Successfully restored raw data file. Rows: {len(pd.read_csv(raw_data_path))}")
else:
    print("❌ Failed to restore data.")
