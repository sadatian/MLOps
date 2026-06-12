# %% [markdown]
# # gRPC Serving, Batch Inference & Release Strategies
#
# In production MLOps, deploying models goes beyond setting up a simple FastAPI endpoint. We must optimize communication protocols, throughput, and deployment safety:
#
# ### High-Performance serving & Release Architectures
#
# *   **gRPC vs REST:** REST APIs serialize data to plain JSON text over HTTP/1.1. gRPC uses binary serialization (Protocol Buffers) over HTTP/2, reducing payload sizes and allowing multiplexing of concurrent requests over a single TCP connection.
# *   **Micro-Batching:** Vectorized inference runs multiple samples concurrently, taking advantage of GPU/CPU cache alignment. A micro-batcher gathers requests over a small window and runs them in a single batch, improving serving throughput.
# *   **Release Engineering:** We split traffic to mitigate risk:
#     *   *Canaries* route a small percentage of user requests to a new model to verify system health.
#     *   *Shadow deployments* duplicate 100% of production traffic to new versions in the background, logging outputs while returning baseline predictions to active users.
#
# ```mermaid
# graph TD
#     subgraph 1. Communication Protocols
#         A[Client HTTP/1.1] -->|Text payload JSON| B[REST API FastAPI]
#         C[Client HTTP/2] -->|Binary serialized Protobuf| D[gRPC Server Handler]
#     end
# 
#     subgraph 2. Micro-Batching Scheduler
#         E[Single request input] -->|Enqueue| F[Request Queue]
#         F -->|Batch window: max wait OR max size| G[MicroBatcher Thread]
#         G -->|Vectorized Batch predict| H[ML Model Engine]
#         H -->|Distribute predictions| I[Resolve await loops]
#     end
# 
#     subgraph 3. Traffic Release Strategy
#         J[Active User Requests] --> K[Routing Gateway Proxy]
#         K -->|90% Canary Route| L[Model Baseline A]
#         K -->|10% Canary Route| M[Model Challenger B]
#         
#         K -->|Mirror Background Call| N[Model Shadow B]
#         N -.->|Log & compare outputs| O[(Evaluation Metrics DB)]
#     end
# 
#     style D fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#     style H fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#     style N fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px
# ```
#
# In this module, we will explore:
# 1. **gRPC vs. REST:** REST (via JSON over HTTP/1.1) is easy to debug but human-readable text serialization has high latency and payload overhead. gRPC (using binary Protocol Buffers over HTTP/2) offers multiplexing and binary packing for high-performance, low-latency microservices.
# 2. **Batch Inference:** Running inference request-by-request is inefficient for deep learning or machine learning models. **Micro-batching** aggregates requests over a small time window and runs them concurrently in a single vectorized computation.
# 3. **Release Strategies:** Deploying new models requires risk mitigation. We use **Canary Releases** (traffic splitting), **Shadow Deployments** (dual-piping without production impact), and **A/B Testing** with statistical significance gating to decide when to promote a new model.


# %%
import os
import time
import struct
import random
import pickle
import queue
import threading
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import scipy.stats as stats

# Define dummy target paths for model fallback
MODEL_PATH = "data/model.pkl"

# %% [markdown]
# ## 1. gRPC & Protocol Buffers Simulation
#
# Protocol Buffers (Protobuf) serialize objects into compact binary buffers.
# We will define a simulated `PredictRequest` protobuf mapping to this schema:
#
# ```protobuf
# syntax = "proto3";
# message PredictRequest {
#     float area_sqft = 1;
#     int32 bedrooms = 2;
# }
# ```

# %%
class PredictRequestProto:
    """Simulated Protobuf message serializer/deserializer."""
    def __init__(self, area_sqft: float, bedrooms: int):
        self.area_sqft = area_sqft
        self.bedrooms = bedrooms

    def serialize(self) -> bytes:
        # Tag 1, wire type 5 (32-bit float): (1 << 3) | 5 = 13 (0x0D)
        tag1 = struct.pack("B", 13)
        val1 = struct.pack("<f", self.area_sqft)

        # Tag 2, wire type 0 (varint/int32): (2 << 3) | 0 = 16 (0x10)
        tag2 = struct.pack("B", 16)
        val2 = struct.pack("<i", self.bedrooms)

        return tag1 + val1 + tag2 + val2

    @classmethod
    def deserialize(cls, data: bytes) -> "PredictRequestProto":
        area_sqft = 0.0
        bedrooms = 0
        i = 0
        while i < len(data):
            tag = data[i]
            i += 1
            if tag == 13:
                area_sqft = struct.unpack("<f", data[i:i+4])[0]
                i += 4
            elif tag == 16:
                bedrooms = struct.unpack("<i", data[i:i+4])[0]
                i += 4
        return cls(area_sqft, bedrooms)

# %% [markdown]
# ### JSON vs. Protobuf Benchmark
# Let's compare payload size and serialization/deserialization latency of JSON vs. our Protobuf implementation.

# %%
import json

# Sample payload
sample_json = {"area_sqft": 1850.5, "bedrooms": 3}
proto_msg = PredictRequestProto(1850.5, 3)

# 1. Payload Size Comparison
json_bytes = json.dumps(sample_json).encode("utf-8")
proto_bytes = proto_msg.serialize()

print(f"📊 Payload Size Comparison:")
print(f"  JSON Payload:     {len(json_bytes)} bytes ('{json.dumps(sample_json)}')")
print(f"  Protobuf Payload: {len(proto_bytes)} bytes (hex: {proto_bytes.hex().upper()})")
print(f"  Size Reduction:   {((len(json_bytes) - len(proto_bytes)) / len(json_bytes)) * 100:.1f}%\n")

# 2. Performance Speed Benchmark
iterations = 10000

# JSON benchmark
start_json = time.time()
for _ in range(iterations):
    encoded = json.dumps(sample_json).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))
time_json = time.time() - start_json

# Protobuf benchmark
start_proto = time.time()
for _ in range(iterations):
    encoded = proto_msg.serialize()
    decoded = PredictRequestProto.deserialize(encoded)
time_proto = time.time() - start_proto

print(f"⏱️ Speed Benchmark ({iterations} iterations):")
print(f"  JSON Serialization/Deserialization:     {time_json:.4f}s")
print(f"  Protobuf Serialization/Deserialization: {time_proto:.4f}s")
print(f"  Speedup Factor:                         {time_json / time_proto:.1f}x")

# %% [markdown]
# ## 2. Batch Inference Engine
#
# Individually calculating predictions for sequential requests has high overhead.
# We will implement a `MicroBatcher` class. It queues incoming inference requests and processes them in batches using a vectorized pandas call.

# %%
class MicroBatcher:
    def __init__(self, model: Any, max_batch_size: int = 4, max_wait_sec: float = 0.02):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_sec = max_wait_sec
        self.queue = queue.Queue()
        self.results = {}
        self.running = True
        self.thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.thread.start()

    def _batch_loop(self):
        while self.running:
            batch = []
            start_time = time.time()
            # Aggregate items until max_batch_size is met or timeout is reached
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.max_wait_sec:
                try:
                    # Brief timeout to avoid locking CPU
                    item = self.queue.get(timeout=0.005)
                    batch.append(item)
                except queue.Empty:
                    continue
            
            if batch:
                # Compile requests into a single DataFrame for vectorized prediction
                features = pd.DataFrame([
                    [r["request"].area_sqft / 1000.0, r["request"].bedrooms] for r in batch
                ], columns=["area_k_sqft", "bedrooms"])
                
                # Perform vectorized batch inference
                predictions = self.model.predict(features)
                
                # Set results and trigger completion events
                for r, pred in zip(batch, predictions):
                    self.results[r["id"]] = float(pred)
                    r["event"].set()

    def predict(self, req: PredictRequestProto) -> float:
        req_id = id(req)
        event = threading.Event()
        self.queue.put({"id": req_id, "request": req, "event": event})
        
        # Wait for the batch processing thread to complete the execution
        event.wait()
        return self.results.pop(req_id)

    def shutdown(self):
        self.running = False
        self.thread.join()

# %% [markdown]
# ### Testing Micro-Batching
# Let's mock a model predictor and verify that the micro-batcher returns correct predictions.

# %%
class MockModel:
    def predict(self, df: pd.DataFrame) -> List[float]:
        # Price = area_k_sqft * 100000 + bedrooms * 15000 + 50000
        return (df["area_k_sqft"] * 100000.0 + df["bedrooms"] * 15000.0 + 50000.0).tolist()

mock_model = MockModel()
batcher = MicroBatcher(mock_model, max_batch_size=4, max_wait_sec=0.05)

# Simulate 4 concurrent requests
reqs = [
    PredictRequestProto(1500.0, 3),
    PredictRequestProto(2000.0, 4),
    PredictRequestProto(1200.0, 2),
    PredictRequestProto(2500.0, 4),
]

threads = []
outputs = [None] * len(reqs)

def worker(idx, req):
    outputs[idx] = batcher.predict(req)

for idx, req in enumerate(reqs):
    t = threading.Thread(target=worker, args=(idx, req))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("⚡ Micro-batcher Vectorized Inference complete:")
for idx, req in enumerate(reqs):
    print(f"  Request {idx}: Area {req.area_sqft} sqft, Bedrooms {req.bedrooms} -> Predicted Price: ${outputs[idx]:,.2f}")

# Clean shutdown of microbatch thread
batcher.shutdown()

# %% [markdown]
# ## 3. Traffic Routing: Canary and Shadow Release
#
# During deployments, we route traffic using routing proxies:
#
# - **Canary Router:** Sends a small slice of user traffic to the new model (e.g. 10% canary, 90% baseline).
# - **Shadow Router:** Clones all production traffic to the new model in the background. The new model's output is logged but discarded, ensuring zero risk to active users.

# %%
class RoutingProxy:
    def __init__(self, model_baseline: Any, model_canary: Any):
        self.baseline = model_baseline
        self.canary = model_canary
        self.shadow_predictions = []

    def route_canary(self, req: PredictRequestProto, canary_weight: float = 0.1) -> float:
        """Routes traffic to the canary model with probability canary_weight."""
        features = pd.DataFrame([[req.area_sqft / 1000.0, req.bedrooms]], columns=["area_k_sqft", "bedrooms"])
        if random.random() < canary_weight:
            return float(self.canary.predict(features)[0])
        return float(self.baseline.predict(features)[0])

    def route_shadow(self, req: PredictRequestProto) -> float:
        """
        Routes traffic to the baseline model, while asynchronously running
        inference on the shadow model in the background to log outputs.
        """
        features = pd.DataFrame([[req.area_sqft / 1000.0, req.bedrooms]], columns=["area_k_sqft", "bedrooms"])
        
        # 1. Main route (synchronous return)
        pred_baseline = float(self.baseline.predict(features)[0])
        
        # 2. Shadow route (simulated async/background thread)
        def log_shadow():
            pred_shadow = float(self.canary.predict(features)[0])
            self.shadow_predictions.append({
                "request": req,
                "baseline": pred_baseline,
                "shadow": pred_shadow
            })
            
        t = threading.Thread(target=log_shadow)
        t.start()
        t.join() # Join in simulation for deterministic outcomes
        
        return pred_baseline

# %% [markdown]
# ## 4. A/B Test Significance Gating
#
# When running A/B tests to compare Model A (Baseline) and Model B (Challenger), we measure prediction absolute errors against actual validation data.
# We perform a two-sample t-test to check if the challenger's error reduction is statistically significant ($p < 0.05$).

# %%
def perform_significance_gate(
    errors_baseline: np.ndarray, 
    errors_challenger: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[bool, float, float]:
    """
    Executes a two-sample independent t-test.
    Returns (gate_passed, p_value, mean_error_difference).
    """
    mean_base = np.mean(errors_baseline)
    mean_chal = np.mean(errors_challenger)
    diff = mean_base - mean_chal
    
    # Run independent t-test (one-sided: challenger error is less than baseline error)
    t_stat, p_val = stats.ttest_ind(errors_baseline, errors_challenger, alternative="greater")
    
    # Gate passes if mean challenger error is lower and the result is statistically significant
    gate_passed = (diff > 0) and (p_val < alpha)
    
    return gate_passed, p_val, diff

# %% [markdown]
# ### Simulating A/B Test Gating
# We simulate error distributions for two models. 
# - Scenario 1: Challenger has slightly lower error, but not statistically significant (fail gate).
# - Scenario 2: Challenger has significantly lower error (pass gate).

# %%
np.random.seed(42)
num_samples = 100

# Baseline model errors
errors_base = np.random.normal(25000, 5000, size=num_samples)

# Challenger Scenario 1 (slight change, high noise)
errors_chal_1 = np.random.normal(24500, 5000, size=num_samples)

# Challenger Scenario 2 (strong change)
errors_chal_2 = np.random.normal(22000, 4800, size=num_samples)

# Scenario 1 Gate check
passed1, p1, diff1 = perform_significance_gate(errors_base, errors_chal_1)
print(f"📊 A/B Scenario 1 (Slight, non-significant change):")
print(f"  Error Diff: {diff1:.2f}, p-value: {p1:.4f}")
print(f"  Gate Passed: {passed1} (Do NOT promote model)\n")

# Scenario 2 Gate check
passed2, p2, diff2 = perform_significance_gate(errors_base, errors_chal_2)
print(f"📊 A/B Scenario 2 (Significant change):")
print(f"  Error Diff: {diff2:.2f}, p-value: {p2:.4f}")
print(f"  Gate Passed: {passed2} (PROMOTE model to production)")

# %% [markdown]
# ---
#
# 🎉 **Module 14 Completed!** You have successfully implemented gRPC serialization, vectorized micro-batch serving schedulers, and statistical A/B release gates!
