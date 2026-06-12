from src.module_12_pipeline_orchestration.orchestration_guide import Task, DAG, TaskState
import pytest

def test_topological_sort():
    dag = DAG("Test_Topological")
    t1 = Task("T1", lambda: None)
    t2 = Task("T2", lambda: None)
    t3 = Task("T3", lambda: None)
    t4 = Task("T4", lambda: None)

    # Dependencies: T1 -> T2 -> T4, T1 -> T3 -> T4
    t1 >> t2 >> t4
    t1 >> t3 >> t4

    dag.add_tasks(t1, t2, t3, t4)
    order = [t.name for t in dag.topological_sort()]
    
    assert order[0] == "T1"
    assert order[-1] == "T4"
    assert set(order[1:3]) == {"T2", "T3"}

def test_cycle_detection():
    dag = DAG("Test_Cycle")
    t1 = Task("T1", lambda: None)
    t2 = Task("T2", lambda: None)

    # Circular dependency: T1 -> T2 -> T1
    t1 >> t2 >> t1
    dag.add_tasks(t1, t2)
    
    with pytest.raises(ValueError, match="Cyclical Dependency Detected"):
        dag.validate()

def test_task_retries():
    attempts = 0
    def flaky_task():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Transient error")
        return

    t = Task("Flaky", flaky_task, retries=2, retry_delay=0.01)
    success = t.execute()
    
    assert success is True
    assert t.state == TaskState.SUCCESS
    assert t.remaining_retries == 0
    assert attempts == 3

def test_task_permanent_failure():
    def failing_task():
        raise RuntimeError("Unrecoverable error")

    t = Task("Broken", failing_task, retries=1, retry_delay=0.01)
    success = t.execute()
    
    assert success is False
    assert t.state == TaskState.FAILED
    assert t.remaining_retries == 0

def test_downstream_skip():
    dag = DAG("Test_Skip")
    t1 = Task("T1", lambda: None)
    
    def failing_task():
        raise RuntimeError("Fail")
        
    t2 = Task("T2", failing_task, retries=0)
    t3 = Task("T3", lambda: None)

    # Dependencies: T1 -> T2 -> T3
    t1 >> t2 >> t3
    dag.add_tasks(t1, t2, t3)
    states = dag.execute()

    assert states["T1"] == TaskState.SUCCESS
    assert states["T2"] == TaskState.FAILED
    assert states["T3"] == TaskState.UPSTREAM_FAILED
