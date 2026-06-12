import numpy as np
import pandas as pd
from src.advanced_mlops.module_14_grpc_batch_release.grpc_guide import (
    PredictRequestProto,
    MicroBatcher,
    MockModel,
    RoutingProxy,
    perform_significance_gate
)

def test_protobuf_serialization():
    original = PredictRequestProto(1500.5, 3)
    serialized = original.serialize()
    
    # Payload should be exactly 10 bytes:
    # 1 byte tag + 4 bytes float (area) + 1 byte tag + 4 bytes int (bedrooms)
    assert len(serialized) == 10
    
    deserialized = PredictRequestProto.deserialize(serialized)
    assert deserialized.area_sqft == 1500.5
    assert deserialized.bedrooms == 3

def test_micro_batcher():
    model = MockModel()
    batcher = MicroBatcher(model, max_batch_size=2, max_wait_sec=0.01)
    
    req1 = PredictRequestProto(1000.0, 2)
    req2 = PredictRequestProto(2000.0, 4)
    
    import threading
    results = [None, None]
    
    def run_worker(idx, req):
        results[idx] = batcher.predict(req)
        
    t1 = threading.Thread(target=run_worker, args=(0, req1))
    t2 = threading.Thread(target=run_worker, args=(1, req2))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Check predictions:
    # req1: 1.0 * 100000 + 2 * 15000 + 50000 = 180000.0
    # req2: 2.0 * 100000 + 4 * 15000 + 50000 = 310000.0
    assert results[0] == 180000.0
    assert results[1] == 310000.0
    
    batcher.shutdown()

def test_routing_proxy():
    model_base = MockModel()
    model_canary = MockModel() # can use same class for simulation
    
    proxy = RoutingProxy(model_base, model_canary)
    req = PredictRequestProto(1500.0, 3)
    
    # Test canary routing (extreme weights to verify logic path)
    res_base = proxy.route_canary(req, canary_weight=0.0)
    res_canary = proxy.route_canary(req, canary_weight=1.0)
    
    # For area 1500, bedrooms 3: 1.5 * 100000 + 3 * 15000 + 50000 = 245000.0
    assert res_base == 245000.0
    assert res_canary == 245000.0
    
    # Test shadow routing
    res_shadow = proxy.route_shadow(req)
    assert res_shadow == 245000.0
    assert len(proxy.shadow_predictions) == 1
    assert proxy.shadow_predictions[0]["baseline"] == 245000.0
    assert proxy.shadow_predictions[0]["shadow"] == 245000.0

def test_significance_gate():
    errors_base = np.array([10.0, 12.0, 11.0, 9.0, 10.0] * 10)
    errors_worse = np.array([15.0, 14.0, 16.0, 14.0, 15.0] * 10)
    errors_better = np.array([5.0, 4.0, 5.0, 6.0, 4.0] * 10)
    
    # Challenger worse: should fail gate
    passed1, p1, diff1 = perform_significance_gate(errors_base, errors_worse)
    assert passed1 == False
    assert diff1 < 0
    
    # Challenger better: should pass gate
    passed2, p2, diff2 = perform_significance_gate(errors_base, errors_better)
    assert passed2 == True
    assert diff2 > 0
    assert p2 < 0.05
