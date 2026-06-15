<!-- MLOps Overhauled Home Page -->
<div class="homepage-container" markdown="1">

<div class="custom-hero">
    <span class="hero-badge">Interactive Labs &amp; Code</span>
    <h1>MLOps Orchestration and Engineering</h1>
    <p class="hero-subtitle">Master the engineering foundations of production Machine Learning. Build local simulation environments, implement automated quality gates, and orchestrate containerized microservices step-by-step.</p>
    <div class="hero-buttons">
        <a href="src/setup_warmup/module_01_environment_uv/uv_guide/" class="hero-btn btn-primary">Start First Lab</a>
        <a href="#curriculum-roadmap" class="hero-btn btn-outline">Explore Roadmap</a>
    </div>
</div>

<p class="homepage-intro">
    Welcome to <strong>MLOps Orchestration and Engineering</strong>! This sandbox is designed to bridge the gap between model development and production-grade software engineering. By establishing automated pipelines, robust continuous integration, containerized microservices, and continuous observability, you ensure that machine learning systems are scalable, reproducible, and production-ready.
    Instead of writing standard markdown documentation or clicking through heavy Jupyter Notebooks, all tutorials in this project are written as fully executable, highly-celled Python (<code>.py</code>) scripts structured in the Jupytext percent format (<code># %%</code>). This allows you to run them cell-by-cell in your favorite IDE (VS Code, PyCharm, etc.), while compiling into beautiful notebooks on this website.
</p>

---

## 🎯 Core Pillars & Tracks

<div class="feature-grid">
    <div class="feature-card">
        <div class="card-icon">⚙️</div>
        <h3>1. Infrastructure &amp; Environments</h3>
        <p>Establish fully reproducible, local-first environments and simulated cloud services on your workstation.</p>
        <ul>
            <li><strong>Package Management:</strong> Synchronize deterministic virtual environments using ultra-fast <code>uv</code> lockfiles.</li>
            <li><strong>Jupytext Pipeline:</strong> Author fully executable, cell-structured Python (<code>.py</code>) files that automatically compile into clean documentation.</li>
            <li><strong>Simulated Cloud Storage:</strong> Mock model registries and dataset buckets locally utilizing <code>boto3</code> and <code>moto</code> server contexts.</li>
            <li><strong>Infrastructure as Code:</strong> Provision local assets and resources declaratively using <code>Terraform</code> with compliance policies.</li>
            <li><strong>Hardware Portability:</strong> Abstract compute layers with automated <code>CUDA</code> GPU availability checks and CPU fallback routing.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">🧪</div>
        <h3>2. ML Lifecycle &amp; Pipelines</h3>
        <p>Orchestrate multi-stage pipelines, track experiments, and version dataset lineages without git bloat.</p>
        <ul>
            <li><strong>Data Version Control:</strong> Track large dataset pointers and metadata off-git using <code>DVC</code> to version model assets.</li>
            <li><strong>Experiment Tracking:</strong> Programmatically log hyperparameters, training metrics, and registered models using <code>MLflow</code>.</li>
            <li><strong>Reproducible Pipelines:</strong> Build multi-step cached pipeline DAGs using <code>dvc.yaml</code> to avoid redundant compute.</li>
            <li><strong>Workflow Orchestration:</strong> Construct Directed Acyclic Graphs (DAGs) using topological sorting for run execution.</li>
            <li><strong>DAG Fault Tolerance:</strong> Implement custom run retry delays, upstream task skipping, and validation cycles.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">🚀</div>
        <h3>3. Serving &amp; Containers</h3>
        <p>Deploy trained weights behind fast API interfaces, build optimized containers, and manage releases safely.</p>
        <ul>
            <li><strong>REST Inference APIs:</strong> Serve low-latency predictions via <code>FastAPI</code> with strict input/output Pydantic schemas.</li>
            <li><strong>gRPC Microservices:</strong> Implement high-throughput binary serialization using <code>Protocol Buffers</code> and <code>gRPC</code>.</li>
            <li><strong>Vectorized Micro-Batching:</strong> Queue concurrent REST/gRPC requests into optimized batched inference arrays.</li>
            <li><strong>Dockerization:</strong> Compile minimal, secure Alpine/Debian container runners using multi-stage Docker builds.</li>
            <li><strong>Advanced Release Strategies:</strong> Safely split traffic using canary releases, shadow deployments, and A/B test gates.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">📊</div>
        <h3>4. CI/CD &amp; Observability</h3>
        <p>Validate regression guardrails automatically in pull requests and monitor performance drift in production.</p>
        <ul>
            <li><strong>CI/ML Quality Gates:</strong> Enforce programmatic unit tests checking model RMSE and R2 bounds before container compilation.</li>
            <li><strong>CI/CD Pipelines:</strong> Automate unit testing, code linting, and image builds using containerized <code>GitHub Actions</code> workflow loops.</li>
            <li><strong>Statistical Monitoring:</strong> Detect covariate shift and feature/prediction distribution drift utilizing <code>Evidently AI</code>.</li>
            <li><strong>Continuous Training (CT):</strong> Set up automated MLflow retraining triggers linked to live concept drift observations.</li>
            <li><strong>LLMOps Observability:</strong> Log prompts, trace hierarchical spans, and monitor token usage and response latencies.</li>
        </ul>
    </div>
</div>

---

## 🧭 Curriculum Roadmap

<div class="roadmap-section" id="curriculum-roadmap">
    
    <div class="roadmap-grid">
        <!-- Module 1 -->
        <a href="src/setup_warmup/module_01_environment_uv/uv_guide/" class="roadmap-item">
            <div class="roadmap-num">⚙️</div>
            <div class="roadmap-content">
                <h4>Modern Python Environment &amp; Dependency Control with <code>uv</code></h4>
                <p>Lightning-fast virtual environments and synchronized workspace lockfiles.</p>
            </div>
        </a>
        <!-- Module 2 -->
        <a href="src/setup_warmup/module_02_documentation/doc_guide/" class="roadmap-item">
            <div class="roadmap-num">📝</div>
            <div class="roadmap-content">
                <h4>Notebook Documentation with <code>Jupytext</code> &amp; <code>MkDocs</code></h4>
                <p>Execute scripts natively in cells and auto-compile them into clean web docs.</p>
            </div>
        </a>
        <!-- Module 3 -->
        <a href="src/setup_warmup/module_03_aws_simulation/cloud_sims/" class="roadmap-item">
            <div class="roadmap-num">☁️</div>
            <div class="roadmap-content">
                <h4>Cloud Services Simulations and Mock Servers</h4>
                <p>Interact with S3 locally inside python cells using Moto context mocks.</p>
            </div>
        </a>
        <!-- Module 4 -->
        <a href="src/data_experimentation/module_04_data_versioning_dvc/dvc_guide/" class="roadmap-item">
            <div class="roadmap-num">📦</div>
            <div class="roadmap-content">
                <h4>Data Versioning with <code>DVC</code></h4>
                <p>Track large datasets and store metadata outside of Git repositories.</p>
            </div>
        </a>
        <!-- Module 5 -->
        <a href="src/data_experimentation/module_05_experiment_tracking_mlflow/mlflow_guide/" class="roadmap-item">
            <div class="roadmap-num">🧪</div>
            <div class="roadmap-content">
                <h4>Experiment Tracking &amp; Model Registry with <code>MLflow</code></h4>
                <p>Log metrics, trace hyper-parameters, and register trained models in S3.</p>
            </div>
        </a>
        <!-- Module 6 -->
        <a href="src/data_experimentation/module_06_integrated_pipeline/run_pipeline/" class="roadmap-item">
            <div class="roadmap-num">🔗</div>
            <div class="roadmap-content">
                <h4>Integrated MLOps Pipeline (<code>DVC</code> + <code>MLflow</code>)</h4>
                <p>Trigger structured pipeline runs linking DVC tracked assets and MLflow logs.</p>
            </div>
        </a>
        <!-- Module 7 -->
        <a href="src/model_serving/module_07_model_serving/serve_api/" class="roadmap-item">
            <div class="roadmap-num">🚀</div>
            <div class="roadmap-content">
                <h4>Model Serving API with <code>FastAPI</code></h4>
                <p>Package model inference scripts behind lightweight REST API JSON requests.</p>
            </div>
        </a>
        <!-- Module 8 -->
        <a href="src/model_serving/module_08_containerization/docker_containerization/" class="roadmap-item">
            <div class="roadmap-num">🐳</div>
            <div class="roadmap-content">
                <h4>Model Deployment &amp; Containerization with <code>Docker</code></h4>
                <p>Compile a secure Docker image for the prediction service and run checks.</p>
            </div>
        </a>
        <!-- Module 9 -->
        <a href="src/automation_observability/module_09_model_monitoring/drift_detection/" class="roadmap-item">
            <div class="roadmap-num">📈</div>
            <div class="roadmap-content">
                <h4>Model Monitoring &amp; Data Drift Detection with <code>evidently</code></h4>
                <p>Generate HTML dashboards and identify feature/prediction distribution shifts.</p>
            </div>
        </a>
        <!-- Module 10 -->
        <a href="src/automation_observability/module_10_ci_ml_automation/ci_ml_guide/" class="roadmap-item">
            <div class="roadmap-num">🛡️</div>
            <div class="roadmap-content">
                <h4>Continuous Integration for Machine Learning (CI/ML)</h4>
                <p>Enforce programmatic tests to flag metrics regression before release.</p>
            </div>
        </a>
        <!-- Module 11 -->
        <a href="src/automation_observability/module_11_github_actions/github_actions_guide/" class="roadmap-item">
            <div class="roadmap-num">🤖</div>
            <div class="roadmap-content">
                <h4><code>GitHub Actions</code> CI Pipeline for MLOps</h4>
                <p>Orchestrate verification, data preparation, pipeline runs, and docker builds.</p>
            </div>
        </a>
        <!-- Module 12 -->
        <a href="src/automation_observability/module_12_pipeline_orchestration/orchestration_guide/" class="roadmap-item">
            <div class="roadmap-num">🧭</div>
            <div class="roadmap-content">
                <h4>Pipeline Orchestration &amp; DAGs (<code>Airflow</code> / <code>Prefect</code>)</h4>
                <p>Simulate production orchestrators like Airflow and Prefect with DAG run retry configurations.</p>
            </div>
        </a>
        <!-- Module 13 -->
        <a href="src/advanced_mlops/module_13_feature_store/feast_guide/" class="roadmap-item">
            <div class="roadmap-num">🗄️</div>
            <div class="roadmap-content">
                <h4>Feature Store Implementation (Simulated <code>Feast</code>)</h4>
                <p>Avoid train-serve skew and query online/offline databases for time-travel features.</p>
            </div>
        </a>
        <!-- Module 14 -->
        <a href="src/advanced_mlops/module_14_grpc_batch_release/grpc_guide/" class="roadmap-item">
            <div class="roadmap-num">⚡</div>
            <div class="roadmap-content">
                <h4><code>gRPC</code> Serving, Batch Inference &amp; Release Strategies</h4>
                <p>Compare REST and gRPC protocols, and run canary/shadow deployments.</p>
            </div>
        </a>
        <!-- Module 15 -->
        <a href="src/advanced_mlops/module_15_continuous_training/ct_guide/" class="roadmap-item">
            <div class="roadmap-num">🔄</div>
            <div class="roadmap-content">
                <h4>Continuous Training (<code>CT</code>) &amp; <code>HITL</code> Fallbacks</h4>
                <p>Setup automated retraining schedules with human-in-the-loop fallback overrides.</p>
            </div>
        </a>
        <!-- Module 16 -->
        <a href="src/advanced_mlops/module_16_llmops/llmops_guide/" class="roadmap-item">
            <div class="roadmap-num">🧠</div>
            <div class="roadmap-content">
                <h4><code>LLMOps</code> &amp; Generative AI Pipelines</h4>
                <p>Trace prompts, evaluate RAG quality metrics, and control LLM token costs/latency.</p>
            </div>
        </a>
        <!-- Module 17 -->
        <a href="src/advanced_mlops/module_17_iac_advanced_containers/iac_guide/" class="roadmap-item">
            <div class="roadmap-num">🛠️</div>
            <div class="roadmap-content">
                <h4>Infrastructure as Code (<code>IaC</code>) &amp; Advanced Containerization</h4>
                <p>Provision resources declaratively and build GPU-accelerated runners.</p>
            </div>
        </a>
        <!-- Module 18 -->
        <a href="src/advanced_mlops/module_18_agile_lifecycle/agile_guide/" class="roadmap-item">
            <div class="roadmap-num">📊</div>
            <div class="roadmap-content">
                <h4>(Upcoming) Agile MLOps Lifecycle &amp; Heuristic Baselines</h4>
                <p>Draft sprint metrics, deploy simple rule-based heuristics, and manage project scope.</p>
            </div>
        </a>
    </div>
</div>

---

## 🛠️ Local Environment Quick Start

To run any tutorial module code locally on your Linux or WSL distribution:

<div class="terminal-window">
    <div class="terminal-header">
        <div class="terminal-buttons">
            <span class="terminal-btn close"></span>
            <span class="terminal-btn minimize"></span>
            <span class="terminal-btn maximize"></span>
        </div>
        <div class="terminal-title">bash</div>
        <div></div>
    </div>
    <div class="terminal-body">
        <span class="terminal-comment"># 1. Clone the repository and navigate into the workspace</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">git clone https://github.com/sadatian/MLOps.git &amp;&amp; cd MLOps</span><br><br>
        <span class="terminal-comment"># 2. Synchronize python dependencies inside virtual environment</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv sync</span><br><br>
        <span class="terminal-comment"># 3. Execute any interactive percent-celled python tutorial</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv run python src/setup_warmup/module_01_environment_uv/uv_guide.py</span>
    </div>
</div>
</div>
