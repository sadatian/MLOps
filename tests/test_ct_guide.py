import numpy as np
import pandas as pd
from src.advanced_mlops.module_15_continuous_training.ct_guide import (
    detect_covariate_shift,
    detect_concept_drift,
    retrain_and_register_model,
    HeuristicBaselineModel,
    SafeServingWrapper,
    hitl_approval_gate
)

def test_detect_covariate_shift():
    np.random.seed(42)
    ref = pd.DataFrame({"area_sqft": np.random.normal(1800, 100, size=100)})
    cur_no_shift = pd.DataFrame({"area_sqft": np.random.normal(1800, 100, size=100)})
    cur_with_shift = pd.DataFrame({"area_sqft": np.random.normal(2400, 100, size=100)})
    
    res_no = detect_covariate_shift(ref, cur_no_shift, ["area_sqft"])
    res_yes = detect_covariate_shift(ref, cur_with_shift, ["area_sqft"])
    
    assert res_no["drift_detected"] is False
    assert res_yes["drift_detected"] is True

def test_detect_concept_drift():
    ref_errors = np.random.normal(5000, 1000, size=100)
    cur_no_drift = np.random.normal(5000, 1000, size=100)
    cur_with_drift = np.random.normal(25000, 1000, size=100)
    
    res_no = detect_concept_drift(ref_errors, cur_no_drift)
    res_yes = detect_concept_drift(ref_errors, cur_with_drift)
    
    assert res_no["drift_detected"] is False
    assert res_yes["drift_detected"] is True

def test_retrain_and_register_model():
    np.random.seed(42)
    train_df = pd.DataFrame({
        "area_sqft": np.random.normal(1800, 300, size=50),
        "bedrooms": np.random.choice([2, 3, 4], size=50),
        "price_usd": np.random.normal(300000, 50000, size=50)
    })
    
    res = retrain_and_register_model(train_df, "TestHousingPriceCTModel")
    assert "model" in res
    assert "mae" in res
    assert "r2" in res
    assert "model_uri" in res
    assert res["mae"] > 0

def test_safe_serving_wrapper():
    # Setup dummy model
    class DummyModel:
        def predict(self, X):
            return np.array([200000.0] * len(X))
            
    ml_model = DummyModel()
    heuristic = HeuristicBaselineModel(price_per_sqft=150.0)
    wrapper = SafeServingWrapper(ml_model, heuristic)
    
    # Check predictions for typical query
    df_valid = pd.DataFrame([{"area_sqft": 1500.0, "bedrooms": 3}])
    preds = wrapper.predict(df_valid)
    assert preds[0] == 200000.0
    
    # Check predictions for OOD query (should fallback to heuristic)
    df_ood = pd.DataFrame([{"area_sqft": 15000.0, "bedrooms": 3}])
    preds_ood = wrapper.predict(df_ood)
    # 15000 * 150.0 = 2250000.0
    assert preds_ood[0] == 2250000.0
    
    # Check fallback when activated globally
    wrapper.fallback_active = True
    preds_fallback = wrapper.predict(df_valid)
    # 1500 * 150.0 = 225000.0
    assert preds_fallback[0] == 225000.0

def test_hitl_approval_gate():
    # Force approve
    assert hitl_approval_gate("TestModel", 1000, 500, force_approve=True) is True
    # Non-interactive fallback: new_mae < old_mae -> approve
    assert hitl_approval_gate("TestModel", 1000, 500, force_approve=False) is True
    assert hitl_approval_gate("TestModel", 500, 1000, force_approve=False) is False
