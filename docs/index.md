<!-- MLOps Local Dev Sandbox Overhauled Home Page -->

<div class="custom-hero">
    <span class="hero-badge">Interactive Labs &amp; Code</span>
    <h1>MLOps Local Dev Sandbox</h1>
    <p class="hero-subtitle">Master the engineering foundations of production Machine Learning. Build local simulation environments, implement automated quality gates, and orchestrate containerized microservices step-by-step.</p>
    <div class="hero-buttons">
        <a href="src/setup_warmup/module_01_environment_uv/uv_guide/" class="hero-btn btn-primary">Start First Lab</a>
        <a href="#curriculum-roadmap" class="hero-btn btn-outline">Explore Roadmap</a>
    </div>
</div>
Welcome to the **MLOps Local Dev Sandbox** repository. This is a hands-on learning sandbox designed to teach key production Machine Learning Operations concepts step-by-step.

Instead of writing standard markdown documentation or clicking through heavy Jupyter Notebooks, all tutorials in this project are written as **fully executable, highly-celled Python (`.py`) scripts** structured in the Jupytext percent format (`# %%`). This allows you to run them cell-by-cell in your favorite IDE (VS Code, PyCharm, etc.), while compiling into beautiful notebooks on this website.

---

## 🎯 Core Pillars & Tracks

<div class="feature-grid">
    <div class="feature-card">
        <div class="card-icon">⚙️</div>
        <h3>1. Infrastructure &amp; Environments</h3>
        <p>Establish fully reproducible workspaces and local AWS simulation pipelines.</p>
        <ul>
            <li><strong>Environment:</strong> Ultra-fast environment sync via <code>uv</code>.</li>
            <li><strong>Notebooks:</strong> Document execution layouts via Jupytext.</li>
            <li><strong>AWS Mocking:</strong> Standalone local S3 setups using <code>moto</code>.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">🧪</div>
        <h3>2. ML Lifecycle &amp; Pipelines</h3>
        <p>Version datasets outside of Git, track training metrics, and run registry storage jobs.</p>
        <ul>
            <li><strong>Tracking:</strong> Log metrics, params, and weights with MLflow.</li>
            <li><strong>Data Versioning:</strong> Local cloud mimicking using DVC.</li>
            <li><strong>DVC Pipelines:</strong> Link DAG stages through <code>dvc.yaml</code>.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">🚀</div>
        <h3>3. Serving &amp; Containers</h3>
        <p>Wrap inference routines in web microservices and compile production-grade containers.</p>
        <ul>
            <li><strong>Inference APIs:</strong> Package prediction logic with FastAPI.</li>
            <li><strong>Docker:</strong> Compile lightweight, secure container images.</li>
            <li><strong>Testing Gates:</strong> Enforce code-level validation checks.</li>
        </ul>
    </div>
    <div class="feature-card">
        <div class="card-icon">📊</div>
        <h3>4. CI/CD &amp; Observability</h3>
        <p>Run automated quality pipelines and observe model data health in production.</p>
        <ul>
            <li><strong>Drift Analysis:</strong> Evident AI feature shift reports.</li>
            <li><strong>Automation:</strong> GitHub Actions verification triggers.</li>
            <li><strong>Quality Gates:</strong> Prevent regression with performance gates.</li>
        </ul>
    </div>
</div>

---

<div class="roadmap-section" id="curriculum-roadmap">
    <h3>🧭 Curriculum Roadmap</h3>
    <div class="roadmap-progress">
        <span style="font-size: 0.9rem; font-weight: bold; color: #7d2a44;">Progress: 11 / 18 Modules Complete</span>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: 61%;"></div>
        </div>
        <span style="font-size: 0.85rem; color: #71717a; font-weight: 600;">61% Done</span>
    </div>
    
    <div class="roadmap-grid">
        <!-- Module 1 -->
        <div class="roadmap-item">
            <div class="roadmap-num">⚙️</div>
            <div class="roadmap-content">
                <h4>Environment Management (uv) <span class="status-badge completed">completed</span></h4>
                <p>Lightning-fast virtual environments and synchronized workspace lockfiles.</p>
            </div>
        </div>
        <!-- Module 2 -->
        <div class="roadmap-item">
            <div class="roadmap-num">📝</div>
            <div class="roadmap-content">
                <h4>Notebook Documentation (Jupytext) <span class="status-badge completed">completed</span></h4>
                <p>Execute scripts natively in cells and auto-compile them into clean web docs.</p>
            </div>
        </div>
        <!-- Module 3 -->
        <div class="roadmap-item">
            <div class="roadmap-num">☁️</div>
            <div class="roadmap-content">
                <h4>AWS S3 Simulation (Moto) <span class="status-badge completed">completed</span></h4>
                <p>Interact with S3 locally inside python cells using Moto context mocks.</p>
            </div>
        </div>
        <!-- Module 4 -->
        <div class="roadmap-item">
            <div class="roadmap-num">📦</div>
            <div class="roadmap-content">
                <h4>Data Versioning (DVC) <span class="status-badge completed">completed</span></h4>
                <p>Track large datasets and store metadata outside of Git repositories.</p>
            </div>
        </div>
        <!-- Module 5 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🧪</div>
            <div class="roadmap-content">
                <h4>Experiment Tracking (MLflow) <span class="status-badge completed">completed</span></h4>
                <p>Log metrics, trace hyper-parameters, and register trained models in S3.</p>
            </div>
        </div>
        <!-- Module 6 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🔗</div>
            <div class="roadmap-content">
                <h4>Integrated MLOps Pipeline <span class="status-badge completed">completed</span></h4>
                <p>Trigger structured pipeline runs linking DVC tracked assets and MLflow logs.</p>
            </div>
        </div>
        <!-- Module 7 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🚀</div>
            <div class="roadmap-content">
                <h4>Model Serving API (FastAPI) <span class="status-badge completed">completed</span></h4>
                <p>Package model inference scripts behind lightweight REST API JSON requests.</p>
            </div>
        </div>
        <!-- Module 8 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🐳</div>
            <div class="roadmap-content">
                <h4>Docker Containerization <span class="status-badge completed">completed</span></h4>
                <p>Compile a secure Docker image for the prediction service and run checks.</p>
            </div>
        </div>
        <!-- Module 9 -->
        <div class="roadmap-item">
            <div class="roadmap-num">📈</div>
            <div class="roadmap-content">
                <h4>Model Monitoring (Evidently) <span class="status-badge completed">completed</span></h4>
                <p>Generate HTML dashboards and identify feature/prediction distribution shifts.</p>
            </div>
        </div>
        <!-- Module 10 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🛡️</div>
            <div class="roadmap-content">
                <h4>CI/ML Quality Gates <span class="status-badge completed">completed</span></h4>
                <p>Enforce programmatic tests to flag metrics regression before release.</p>
            </div>
        </div>
        <!-- Module 11 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🤖</div>
            <div class="roadmap-content">
                <h4>GitHub Actions CI Pipeline <span class="status-badge completed">completed</span></h4>
                <p>Orchestrate verification, data preparation, pipeline runs, and docker builds.</p>
            </div>
        </div>
        <!-- Module 12 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🧭</div>
            <div class="roadmap-content">
                <h4>Pipeline Orchestration &amp; DAGs <span class="status-badge upcoming">upcoming</span></h4>
                <p>Simulate production orchestrators like Airflow and Prefect with DAG run retry configurations.</p>
            </div>
        </div>
        <!-- Module 13 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🗄️</div>
            <div class="roadmap-content">
                <h4>Feature Store (Feast) <span class="status-badge upcoming">upcoming</span></h4>
                <p>Avoid train-serve skew and query online/offline databases for time-travel features.</p>
            </div>
        </div>
        <!-- Module 14 -->
        <div class="roadmap-item">
            <div class="roadmap-num">⚡</div>
            <div class="roadmap-content">
                <h4>gRPC Serving &amp; Release Strategies <span class="status-badge upcoming">upcoming</span></h4>
                <p>Compare REST and gRPC protocols, and run canary/shadow deployments.</p>
            </div>
        </div>
        <!-- Module 15 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🔄</div>
            <div class="roadmap-content">
                <h4>Continuous Retraining &amp; HITL <span class="status-badge upcoming">upcoming</span></h4>
                <p>Setup automated retraining schedules with human-in-the-loop fallback overrides.</p>
            </div>
        </div>
        <!-- Module 16 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🧠</div>
            <div class="roadmap-content">
                <h4>LLMOps &amp; Generative AI <span class="status-badge upcoming">upcoming</span></h4>
                <p>Trace prompts, evaluate RAG quality metrics, and control LLM token costs/latency.</p>
            </div>
        </div>
        <!-- Module 17 -->
        <div class="roadmap-item">
            <div class="roadmap-num">🛠️</div>
            <div class="roadmap-content">
                <h4>IaC &amp; Advanced Containerization <span class="status-badge upcoming">upcoming</span></h4>
                <p>Provision resources declaratively and build GPU-accelerated runners.</p>
            </div>
        </div>
        <!-- Module 18 -->
        <div class="roadmap-item">
            <div class="roadmap-num">📊</div>
            <div class="roadmap-content">
                <h4>Agile MLOps Lifecycle <span class="status-badge upcoming">upcoming</span></h4>
                <p>Draft sprint metrics, deploy simple rule-based heuristics, and manage project scope.</p>
            </div>
        </div>
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
        <span class="terminal-comment"># 1. Verify that the uv package manager is installed</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv --version</span><br><br>
        <span class="terminal-comment"># 2. Synchronize python dependencies inside virtual environment</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv sync</span><br><br>
        <span class="terminal-comment"># 3. Execute any interactive percent-celled python tutorial</span><br>
        <span class="terminal-prompt">$</span> <span class="terminal-command">uv run python src/setup_warmup/module_01_environment_uv/uv_guide.py</span>
    </div>
</div>
