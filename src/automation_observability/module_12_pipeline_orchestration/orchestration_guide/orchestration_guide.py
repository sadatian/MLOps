# %% [markdown]
# # Pipeline Orchestration & DAGs (Airflow / Prefect)
#
# In production MLOps, machine learning workflows consist of multiple steps: data ingestion, feature validation, model training, and deployment.
# While tools like DVC are great for local, file-based caching and reproducing pipelines on a single workspace, enterprise systems require **orchestration engines** (e.g., Apache Airflow, Prefect, Kubeflow) to manage scheduling, parallel executions, distributed resource routing, state monitoring, and robust error recovery.
#
# In this module, we will explore:
# 1. **DVC vs. Enterprise Orchestrators:** When to use which.
# 2. **DAG Mechanics:** Constructing a Directed Acyclic Graph (DAG) in Python.
# 3. **Topological Sorting & Cycle Detection:** Ensuring execution order and preventing infinite loops.
# 4. **State Management & Retries:** Simulating transient failures, retries, and cascading failures (`UPSTREAM_FAILED`).
# 5. **Real-world Translation:** Comparing our simulation with Apache Airflow and Prefect.

# %% [markdown]
# ## 1. DVC vs. Enterprise Orchestration
#
# | Feature | Local Pipeline (DVC) | Enterprise Orchestrator (Airflow / Prefect) |
# | --- | --- | --- |
# | **Execution Driver** | File changes (Data/Code inputs vs. outputs). | Time schedules, API requests, webhooks, or event streams. |
# | **State Management** | Local `.dvc` tracking and git-hashes. | Central database tracking tasks across multiple workers/nodes. |
# | **Failure Handling** | Fails immediately. No automated retries. | Built-in retry queues, back-off delays, alerting, and manual reruns from point of failure. |
# | **Compute Context** | Single workspace/node (where `dvc repro` runs). | Distributed runners (Kubernetes pods, Celery workers, serverless tasks). |
#
# Let's build a simulated orchestrator engine to understand these execution mechanics under the hood.

# %%
import time
import logging
from typing import Callable, List, Set, Dict, Optional
from enum import Enum

# Configure logger for clear visualization
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")

class TaskState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    UP_FOR_RETRY = "UP_FOR_RETRY"
    FAILED = "FAILED"
    UPSTREAM_FAILED = "UPSTREAM_FAILED"

# %% [markdown]
# ## 2. Core Engine Classes: Task & DAG
#
# We will define a `Task` class representing a single node in a workflow. 
# In orchestrators like Airflow, the shift operators `>>` (downstream) and `<<` (upstream) are overloaded to establish relationships between tasks. We will implement this standard operator overloading.
#
# We will also implement a `DAG` class which manages:
# - Task registration.
# - Cycle validation (ensuring the graph remains acyclic).
# - Topological sorting (finding the correct linear execution order).
# - Sequential execution with error handling, retries, and cascade failure prevention.

# %%
class Task:
    def __init__(
        self,
        name: str,
        func: Callable[[], None],
        retries: int = 0,
        retry_delay: float = 0.05
    ):
        self.name = name
        self.func = func
        self.retries = retries
        self.retry_delay = retry_delay
        self.remaining_retries = retries
        self.upstream: Set["Task"] = set()
        self.downstream: Set["Task"] = set()
        self.state: TaskState = TaskState.PENDING

    def __repr__(self) -> str:
        return f"<Task: {self.name} ({self.state.value})>"

    def __rshift__(self, other: "Task") -> "Task":
        """Overloads the >> operator: self >> other means self runs before other."""
        self.downstream.add(other)
        other.upstream.add(self)
        return other

    def __lshift__(self, other: "Task") -> "Task":
        """Overloads the << operator: self << other means other runs before self."""
        self.upstream.add(other)
        other.downstream.add(self)
        return self

    def execute(self) -> bool:
        """Executes the task, handling retries internally. Returns True if success, else False."""
        self.state = TaskState.RUNNING
        logger.info(f"🚀 Running task: '{self.name}'")
        
        while True:
            try:
                self.func()
                self.state = TaskState.SUCCESS
                logger.info(f"✅ Task '{self.name}' completed successfully.")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Task '{self.name}' failed: {e}")
                if self.remaining_retries > 0:
                    self.state = TaskState.UP_FOR_RETRY
                    logger.info(f"🔄 Retrying '{self.name}' in {self.retry_delay}s... (Retries left: {self.remaining_retries})")
                    self.remaining_retries -= 1
                    time.sleep(self.retry_delay)
                    self.state = TaskState.RUNNING
                else:
                    self.state = TaskState.FAILED
                    logger.error(f"❌ Task '{self.name}' failed permanently. All retries exhausted.")
                    return False

# %%
class DAG:
    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        """Registers a task in the DAG."""
        if task.name in self.tasks:
            raise ValueError(f"Task with name '{task.name}' already registered in DAG '{self.name}'.")
        self.tasks[task.name] = task

    def add_tasks(self, *tasks: Task) -> None:
        """Helper to register multiple tasks at once."""
        for task in tasks:
            self.add_task(task)

    def validate(self) -> None:
        """Validates that the DAG is acyclic (contains no loops). Throws ValueError if cyclical."""
        visiting = set()
        visited = set()

        def dfs(task: Task):
            if task.name in visiting:
                raise ValueError(f"🚨 Cyclical Dependency Detected! Task '{task.name}' is part of a loop.")
            if task.name not in visited:
                visiting.add(task.name)
                for downstream_task in task.downstream:
                    dfs(downstream_task)
                visiting.remove(task.name)
                visited.add(task.name)

        for task in self.tasks.values():
            if task.name not in visited:
                dfs(task)
        logger.info("🛡️ DAG cycle validation passed. The graph is acyclic.")

    def topological_sort(self) -> List[Task]:
        """Returns tasks in topological execution order using Kahn's algorithm."""
        self.validate()
        
        # Kahn's Algorithm
        in_degree = {task.name: len(task.upstream) for task in self.tasks.values()}
        queue = [task for task in self.tasks.values() if in_degree[task.name] == 0]
        order: List[Task] = []

        while queue:
            # Sort by name to ensure deterministic execution for identical dependency layers
            queue.sort(key=lambda t: t.name)
            current = queue.pop(0)
            order.append(current)

            for downstream_task in current.downstream:
                in_degree[downstream_task.name] -= 1
                if in_degree[downstream_task.name] == 0:
                    queue.append(downstream_task)

        if len(order) != len(self.tasks):
            raise ValueError("Failed to resolve dependencies. The DAG contains cycles.")
        
        return order

    def execute(self) -> Dict[str, TaskState]:
        """Executes tasks in topological order, managing upstream failures."""
        logger.info(f"🏁 Starting DAG execution: '{self.name}'")
        execution_order = self.topological_sort()
        logger.info(f"📐 Execution Order: {[t.name for t in execution_order]}")

        # Reset states
        for task in self.tasks.values():
            task.state = TaskState.PENDING
            task.remaining_retries = task.retries

        for task in execution_order:
            # Check if any upstream task failed
            upstream_failed = False
            for parent in task.upstream:
                if parent.state in [TaskState.FAILED, TaskState.UPSTREAM_FAILED]:
                    upstream_failed = True
                    break

            if upstream_failed:
                task.state = TaskState.UPSTREAM_FAILED
                logger.error(f"⏭️ Skipping task '{task.name}' due to UPSTREAM_FAILED parent.")
                continue

            # Run task
            task.execute()

        results = {name: task.state for name, task in self.tasks.items()}
        logger.info(f"📊 DAG execution finished. Final States: {results}")
        return results

# %% [markdown]
# ## 3. Workflow Simulation: Successful Pipeline & Transient Retries
#
# Let's define a mock training pipeline:
# 1. `ingest_data`: Always succeeds.
# 2. `validate_features`: Fails once transiently (e.g. mock API rate limit), then succeeds.
# 3. `train_model`: Always succeeds.
# 4. `evaluate_model`: Always succeeds.
#
# We will check if the automated retry logic manages to rescue the pipeline.

# %%
# State variables for simulating failures
feature_validation_attempts = 0

def ingest_data():
    print("Reading dataset from mock S3 remote...")

def validate_features():
    global feature_validation_attempts
    feature_validation_attempts += 1
    if feature_validation_attempts < 2:
        raise ConnectionError("Timeout connecting to feature validation service! (Transient failure)")
    print("Features verified: target column exists and schemas match.")

def train_model():
    print("Training XGBoost regression baseline model on training set.")

def evaluate_model():
    print("Evaluation finished: RMSE = 0.124. Logged metrics to MLflow.")

# Instantiate tasks
t1 = Task("ingest_data", ingest_data)
t2 = Task("validate_features", validate_features, retries=2, retry_delay=0.1)
t3 = Task("train_model", train_model)
t4 = Task("evaluate_model", evaluate_model)

# Define dependencies: t1 -> t2 -> t3 -> t4
t1 >> t2 >> t3 >> t4

# Build and execute DAG
dag = DAG("ML_Training_Pipeline")
dag.add_tasks(t1, t2, t3, t4)

# Execute the simulation
final_states = dag.execute()

# %% [markdown]
# ## 4. Workflow Simulation: Permanent Failure with Skipping (Cascading Downstream)
#
# If a task fails permanently (exhausting all retries), the downstream tasks shouldn't run.
# Let's model a scenario where `validate_features` fails completely.

# %%
feature_validation_attempts = 0  # reset counter

def permanent_failure():
    raise ValueError("Missing 'price' target column in raw data! (Unrecoverable validation error)")

t_failed = Task("validate_features", permanent_failure, retries=1, retry_delay=0.05)

# Build a new DAG with the failing task
dag_fail = DAG("Failing_Pipeline")
# Re-instantiate other tasks to avoid state sharing
t_ingest = Task("ingest_data", ingest_data)
t_train = Task("train_model", train_model)
t_eval = Task("evaluate_model", evaluate_model)

# Link dependencies
t_ingest >> t_failed >> t_train >> t_eval

dag_fail.add_tasks(t_ingest, t_failed, t_train, t_eval)
failed_states = dag_fail.execute()

# %% [markdown]
# ## 5. DAG Cycle Detection Test
#
# If a pipeline is defined with circular references (e.g., `A >> B >> C >> A`), the orchestrator should immediately raise an exception during validation. Let's see this in action.

# %%
cycle_dag = DAG("Cyclic_Pipeline")
ta = Task("Task_A", lambda: print("A"))
tb = Task("Task_B", lambda: print("B"))
tc = Task("Task_C", lambda: print("C"))

# Establish a circular dependency: A -> B -> C -> A
ta >> tb >> tc >> ta
cycle_dag.add_tasks(ta, tb, tc)

try:
    logger.info("Validating cyclic pipeline...")
    cycle_dag.validate()
except ValueError as e:
    logger.error(f"Cycle validation successfully caught the circular dependency: {e}")

# %% [markdown]
# ## 6. Syntax Comparison: Real Apache Airflow & Prefect Code
#
# Under the hood, production orchestrators map tasks onto DAG structures similarly to our custom class. Here is how you would configure this in production engines:
#
# ### 🐘 Apache Airflow Syntax
#
# Airflow uses standard Python files to define the DAG layout and tasks. Note the identical use of the `>>` shift operators:
#
# ```python
# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
#
# default_args = {
#     'owner': 'airflow',
#     'retries': 2,
#     'retry_delay': timedelta(minutes=5),
# }
#
# with DAG(
#     'airflow_ml_pipeline',
#     default_args=default_args,
#     schedule_interval='@daily',
#     start_date=datetime(2026, 1, 1),
#     catchup=False,
# ) as dag:
#
#     t1 = PythonOperator(task_id='ingest_data', python_callable=ingest_data)
#     t2 = PythonOperator(task_id='validate_features', python_callable=validate_features)
#     t3 = PythonOperator(task_id='train_model', python_callable=train_model)
#     t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model)
#
#     # Define identical dependency syntax
#     t1 >> t2 >> t3 >> t4
# ```
#
# ### 🦋 Prefect 3.x Syntax
#
# Prefect takes a modern, functional approach. Rather than declaring static DAGs, it compiles DAGs dynamically at runtime using `@flow` and `@task` decorators:
#
# ```python
# from prefect import task, flow
# import time
#
# @task(retries=2, retry_delay_seconds=300)
# def validate_features():
#     # Feature validation logic...
#     pass
#
# @task
# def train_model():
#     # Train model logic...
#     pass
#
# @flow(name="Prefect ML Pipeline")
# def ml_pipeline():
#     # Execution follows Python's native functional order!
#     ingest_data()
#     validate_features()
#     train_model()
#     evaluate_model()
#
# if __name__ == "__main__":
#     ml_pipeline()
# ```
#
# Summary: 
# - Airflow requires **explicit DAG and operator bindings** with standard dependency arrows.
# - Prefect tracks dependencies **implicitly through standard functional calls** using decorators.
# Both approaches compile identical directed acyclic graphs for scheduling and monitoring.
