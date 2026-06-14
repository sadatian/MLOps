# %% [markdown]
# # 🧠 LLMOps & Generative AI Pipelines
#
# Productionizing generative AI pipelines requires specific controls and monitoring:
#
# ### The LLMOps Tracing & Span Lifecycle
#
# Traditional metrics (like RMSE and accuracy) do not apply directly to unstructured generative AI outputs. 
# To evaluate and audit LLM systems, we track executions as a hierarchical tree of **Traces** and **Spans**:
# 1. **Trace:** Represents the complete end-to-end request lifecycle (e.g., a RAG conversation).
# 2. **Spans:** Represent granular sub-tasks executed during the trace (e.g., prompt injection scans, vector database queries, LLM API calls). Each span tracks inputs, outputs, token counts, costs, and execution latency.
#
# ```mermaid
#  graph TD
#      subgraph 1_request_ingestion ["1. Request Ingestion"]
#          A["User Input / Prompt"] -->|"Start Trace"| B["Parent Trace Controller"]
#      end
#
#      subgraph 2_hierarchical_span_executions ["2. Hierarchical Span Executions"]
#          B -->|"Span 1: Filter"| C["Prompt Injection Guard"]
#          C -->|"Pass"| D["Span 2: Retrieve"]
#          D -->|"Retrieve matching docs"| E["Vector Database Query"]
#          E -->|"Inject Context"| F["Span 3: Generate"]
#          F -->|"REST call to local model"| G["Gemma-4 LLM Call"]
#          G -->|"Return token stream"| F
#          F -->|"Span 4: Parse"| H["Response Parser"]
#      end
#
#      subgraph 3_mlflow_observability_logging ["3. MLflow Observability Logging"]
#          H -->|"Close Trace"| I["Compile structured trace.json"]
#          I -->|"mlflow.log_artifact"| J["MLflow Artifact Store (S3)"]
#          C -->|"Log latency & scan tokens"| K["mlflow.log_metrics"]
#          F -->|"Log generation cost & latency"| K
#          K -->|"Write metrics"| L["MLflow SQLite Tracking DB"]
#      end
#
#      style C fill:#fff9c4,stroke:#fbc02d,stroke-width:1.5px
#      style G fill:#bbdefb,stroke:#1976d2,stroke-width:1.5px
#      style L fill:#d4edda,stroke:#28a745,stroke-width:1.5px
#
# ```
#
# In this module, we will explore:
# 1. **Prompt Versioning & Registry:** Prompts are parameters of the LLM pipeline. We version them, register them, and log prompt templates to MLflow.
# 2. **Structured LLMOps Tracing:** In complex agentic RAG structures (like CRAG and Self-RAG), requests flow through multiple decision paths and nodes. To debug and inspect these systems, we trace every node execution as a **Span** under a parent **Trace** tree:
#    - **Trace:** Represents the complete query lifecycle, containing a unique `trace_id`, metadata, and evaluations.
#    - **Span:** Represents an individual unit of work (e.g. an LLM call or search tool call), containing inputs, outputs, latency, tokens, costs, and cache flags.
# 3. **Cost, Latency, and Caching:** Caching responses saves API costs and latency, especially during stateful graph loop retries.
# 4. **Security Safeguards:** Filtering prompt injections before calling the LLM.
# 5. **MLflow Artifact Integration:** We log our prompt version parameters and summary metrics to MLflow, and upload the full structured `trace.json` file as a run artifact for audit trail and regression testing.
#
# In this module, we target the local high-end reasoning model (**gemma-4-12b-it-qat-q4_0.gguf** running with `--cache-type 4q_0`).
#


# %%
import os
import sys
import time
import socket
import json
import uuid
from typing import Dict, Any, Tuple, List
from openai import OpenAI
import mlflow

# Port definitions
API_PORTS = [5055, 7860]

# Set up MLflow experiment
mlflow.set_experiment("LLMOps_Evaluation_Module_16")

# %% [markdown]
# ## 📡 1. Port Verification & LLM Server Setup
#
# To run this module locally, you must have:
# 1. An installation of **oobabooga/text-generation-webui** at `/home/t/textgen-main/` (or similar location).
# 2. The **gemma-4-12b-it-qat-q4_0.gguf** quantized model downloaded and placed into the `models/` directory of the `textgen-main` installation.
#
# We check if the local LLM API server is running on port 5055 or 7860.
# If not, we print instructions showing how to launch the reasoning model:
#
# ```bash
# /home/t/textgen-main/start_linux.sh --model gemma-4-12b-it-qat-q4_0.gguf --api --api-port 5055 --cache-type 4q_0
# ```

# %%
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_active_ports() -> int:
    for port in API_PORTS:
        if is_port_in_use(port):
            return port
    return None

active_port = check_active_ports()
if not active_port:
    if __name__ == "__main__":
        print("❌ Error: No local LLM API server detected on port 5055 or 7860.")
        print("Setup Requirements:")
        print("  1. Ensure oobabooga/textgen is installed at '/home/t/textgen-main'.")
        print("  2. Place the 'gemma-4-12b-it-qat-q4_0.gguf' model in the textgen models directory.")
        print("  3. Boot up the local LLM server using the following command:")
        print("     /home/t/textgen-main/start_linux.sh \\")
        print("                     --model gemma-4-12b-it-qat-q4_0.gguf \\")
        print("                     --cache-type 4q_0 \\")
        print("                     --api --api-port 5055")
        sys.exit(1)
    else:
        active_port = 5055  # Fallback port for imports and testing

# Instantiate OpenAI client pointing locally
client = OpenAI(base_url=f"http://localhost:{active_port}/v1", api_key="dummy")

# %% [markdown]
# ## 📐 2. Structured Tracing Schema
#
# We define our telemetry structures: `Span` (for node-level execution details) and `Trace` (for query-level details).

# %%
class Span:
    """
    Represents an individual unit of execution (e.g. LLM call or tool query) within a Trace.
    """
    def __init__(
        self, 
        name: str, 
        input_data: Any, 
        output_data: Any, 
        latency_sec: float, 
        cost_usd: float, 
        prompt_tokens: int, 
        completion_tokens: int, 
        cached: bool
    ):
        self.span_id = str(uuid.uuid4())
        self.name = name
        self.input_data = input_data
        self.output_data = output_data
        self.latency_sec = latency_sec
        self.cost_usd = cost_usd
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached = cached

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "input": self.input_data,
            "output": self.output_data,
            "latency_sec": self.latency_sec,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached": self.cached
        }


class Trace:
    """
    Represents a full transaction lifecycle tracking execution across multiple spans.
    """
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())
        self.query = query
        self.spans: List[Span] = []
        self.metadata: Dict[str, Any] = {}
        self.evaluations: Dict[str, Any] = {}

    def add_span(self, span: Span):
        self.spans.append(span)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
            "evaluations": self.evaluations
        }

# %% [markdown]
# ## 🗄️ 3. Prompt Versioning Registry
#
# Prompts are registered, versioned, and managed. We track the prompts used in Corrective RAG (CRAG) and Self-RAG nodes.

# %%
PROMPT_REGISTRY = {
    # 1. Document Grader Prompt (CRAG)
    "doc_grader": {
        "v1.0.0": {
            "template": (
                "Determine if the documents contain facts that directly help answer the user question.\n"
                "Question: {question}\nDocuments:\n{documents}\nRelevance (YES/NO):"
            ),
            "description": "Simple binary document grading."
        },
        "v2.0.0": {
            "template": (
                "You are an expert validator. Assess if the documents below contain relevant facts to answer the question.\n"
                "If the text is irrelevant, answer NO. Otherwise answer YES.\n"
                "Question: {question}\nDocuments:\n{documents}\nRelevance (YES/NO):"
            ),
            "description": "Stricter document grader template."
        }
    },
    # 2. Synthesis Prompt (Generator)
    "synthesis": {
        "v1.0.0": {
            "template": "Context:\n{context}\nQuery: {question}\nAnswer the query based only on the context:",
            "description": "Basic synthesis."
        },
        "v2.0.0": {
            "template": (
                "Context:\n{context}\nQuery: {question}\n"
                "Using ONLY facts in the context, answer the query. If you cannot answer it, say 'I do not know'.\n"
                "Answer:"
            ),
            "description": "Hallucination-guarded synthesis."
        }
    },
    # 3. Groundedness check (Self-RAG)
    "groundedness": {
        "v1.0.0": {
            "template": (
                "Check if the answer is grounded in the context.\n"
                "Context: {context}\nAnswer: {generation}\nGrounded (YES/NO):"
            ),
            "description": "Basic groundedness grading."
        },
        "v2.0.0": {
            "template": (
                "Verify if all statements in the answer are supported by the context.\n"
                "If the answer contains any claims not mentioned in the context, reply NO. Otherwise YES.\n"
                "Context: {context}\nAnswer: {generation}\nGrounded (YES/NO):"
            ),
            "description": "Stricter groundedness check."
        }
    },
    # 4. Utility check (Self-RAG)
    "utility": {
        "v1.0.0": {
            "template": (
                "Does the answer resolve the question?\n"
                "Question: {question}\nAnswer: {generation}\nUseful (YES/NO):"
            ),
            "description": "Basic utility grading."
        },
        "v2.0.0": {
            "template": (
                "Determine if the answer directly addresses and resolves the question.\n"
                "If the answer is incomplete, say NO. Otherwise YES.\n"
                "Question: {question}\nAnswer: {generation}\nUseful (YES/NO):"
            ),
            "description": "Stricter utility check."
        }
    }
}

def format_registry_prompt(node_name: str, version: str, **kwargs) -> str:
    if node_name not in PROMPT_REGISTRY:
        raise ValueError(f"Node '{node_name}' not in registry.")
    if version not in PROMPT_REGISTRY[node_name]:
        raise ValueError(f"Version '{version}' not found for node '{node_name}'.")
    template = PROMPT_REGISTRY[node_name][version]["template"]
    return template.format(**kwargs)

# %% [markdown]
# ## ⏱️ 4. Cost & Latency Tracking with Exact Cache
#
# - **API Tracker:** Tracks input/output tokens, computes costs ($15.00/M input, $60.00/M output), and measures latency.
# - **Exact Cache:** Bypasses LLM queries when identical prompt matches.

# %%
INPUT_COST_M = 15.00
OUTPUT_COST_M = 60.00

# Global Response Cache
RESPONSE_CACHE: Dict[str, Tuple[str, int, int]] = {}

def query_llm_with_metrics(
    prompt: str, 
    api_client: OpenAI, 
    use_cache: bool = True
) -> Tuple[str, float, float, bool, int, int]:
    """
    Queries LLM or loads from cache.
    Returns (answer, latency_sec, cost_usd, is_cache_hit, prompt_tokens, completion_tokens).
    """
    if use_cache and prompt in RESPONSE_CACHE:
        answer, p_tokens, c_tokens = RESPONSE_CACHE[prompt]
        return answer, 0.0, 0.00, True, p_tokens, c_tokens

    # Cache Miss
    start_time = time.time()
    try:
        response = api_client.chat.completions.create(
            model="gpt-4o-mini", # maps to local model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        latency = time.time() - start_time
        
        answer = response.choices[0].message.content
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens
        
        cost = (p_tokens * INPUT_COST_M / 1_000_000) + (c_tokens * OUTPUT_COST_M / 1_000_000)
        
        if use_cache:
            RESPONSE_CACHE[prompt] = (answer, p_tokens, c_tokens)
            
        return answer, latency, cost, False, p_tokens, c_tokens
    except Exception as e:
        return f"Error: {e}", 0.0, 0.0, False, 0, 0

# %% [markdown]
# ## 🛡️ 5. Prompt Injection Guard
#
# Simple heuristic validation to detect system guidelines overrides.

# %%
def check_prompt_injection(user_input: str) -> bool:
    blacklisted = [
        "ignore previous instructions",
        "ignore the instructions above",
        "system prompt",
        "override system guidelines",
        "reveal your system prompt"
    ]
    cleaned = user_input.lower()
    return any(phrase in cleaned for phrase in blacklisted)

# %% [markdown]
# ## 📊 6. RAG Evaluation Metrics (Grounded Overlap & LLM-as-a-Judge)

# %%
def compute_faithfulness_overlap(answer: str, context: str) -> float:
    """
    Token-overlap based grounding score.
    """
    import re
    def get_keywords(text: str) -> set:
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {"this", "that", "with", "from", "they", "have", "were", "about", "over"}
        return set(w for w in words if w not in stop_words)

    ans_words = get_keywords(answer)
    ctx_words = get_keywords(context)
    if not ans_words:
        return 1.0
    return len(ans_words.intersection(ctx_words)) / len(ans_words)


def compute_context_recall_overlap(ground_truth: str, context: str) -> float:
    """
    Recalled keywords in context compared to ground truth.
    """
    import re
    def get_keywords(text: str) -> set:
        return set(re.findall(r'\b\w{4,}\b', text.lower()))
    gt_words = get_keywords(ground_truth)
    ctx_words = get_keywords(context)
    if not gt_words:
        return 1.0
    return len(gt_words.intersection(ctx_words)) / len(gt_words)


def llm_judge_grounding(context: str, answer: str, api_client: OpenAI) -> float:
    """
    Asks the reasoning LLM to rate grounding score on a scale 0.0 to 5.0.
    """
    judge_prompt = (
        "You are an AI evaluator checking RAG responses. Compare the context and answer below.\n"
        "Rate how faithful the answer is to the context on a scale from 0.0 (unsupported) to 5.0 (perfectly faithful).\n"
        f"Context: {context}\nAnswer: {answer}\n"
        "Provide your rating in this exact format: 'Score: <float_value>'"
    )
    ans, _, _, _, _, _ = query_llm_with_metrics(judge_prompt, api_client, use_cache=True)
    import re
    match = re.search(r'Score:\s*([0-5](?:\.\d+)?)', ans)
    return float(match.group(1)) if match else 4.0

# %% [markdown]
# ## 🧠 7. Stateful Agentic RAG Simulation Trace
#
# We simulate a workflow graph traversal:
# `Retrieve -> Doc Grader -> Web Search -> Generate -> Groundedness Check -> Utility Check`.
# Telemetry details are accumulated in a structured `Trace` object containing nested node `Spans`.

# %%
def simulate_agentic_rag(
    question: str, 
    context: str, 
    ground_truth: str, 
    prompt_version: str, 
    api_client: OpenAI,
    use_cache: bool = True
) -> Trace:
    """
    Simulates a Corrective RAG (CRAG) & Self-RAG graph traversal,
    recording structured spans for each node execution.
    """
    trace = Trace(question)
    trace.metadata = {
        "prompt_version": prompt_version,
        "model_name": "gemma-4-12b-it-qat-q4_0"
    }

    # 0. Prompt Injection Guard Check
    if check_prompt_injection(question):
        # Record blocked span
        trace.add_span(Span(
            name="prompt_injection_guard",
            input_data=question,
            output_data="BLOCKED_BY_GUARD",
            latency_sec=0.001,
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            cached=False
        ))
        trace.metadata["status"] = "failed_security"
        return trace

    # --- Node 1: Doc Grader ---
    grader_prompt = format_registry_prompt("doc_grader", prompt_version, question=question, documents=context)
    ans_grade, lat, cost, cached, p_tok, c_tok = query_llm_with_metrics(grader_prompt, api_client, use_cache)
    
    trace.add_span(Span(
        name="doc_grader",
        input_data=grader_prompt,
        output_data=ans_grade,
        latency_sec=lat,
        cost_usd=cost,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cached=cached
    ))
    
    is_relevant = "YES" in ans_grade.upper()

    # --- Node 2: Web Search (Simulated CRAG tool node) ---
    working_context = context
    if not is_relevant:
        search_query = f"Search query for: {question}"
        web_search_output = "[Web Search Result] Web results confirm that special operations forces do not operate bases on Mars."
        
        # Record search tool span
        trace.add_span(Span(
            name="web_search",
            input_data=search_query,
            output_data=web_search_output,
            latency_sec=0.15,  # mock network latency
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            cached=False
        ))
        working_context += f"\n{web_search_output}"

    # --- Node 3: Synthesis Generation ---
    synth_prompt = format_registry_prompt("synthesis", prompt_version, question=question, context=working_context)
    ans_gen, lat, cost, cached, p_tok, c_tok = query_llm_with_metrics(synth_prompt, api_client, use_cache)
    
    trace.add_span(Span(
        name="synthesis_generation",
        input_data=synth_prompt,
        output_data=ans_gen,
        latency_sec=lat,
        cost_usd=cost,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cached=cached
    ))

    # --- Node 4: Groundedness Check (Self-RAG) ---
    ground_prompt = format_registry_prompt("groundedness", prompt_version, context=working_context, generation=ans_gen)
    ans_ground, lat, cost, cached, p_tok, c_tok = query_llm_with_metrics(ground_prompt, api_client, use_cache)
    
    trace.add_span(Span(
        name="groundedness_check",
        input_data=ground_prompt,
        output_data=ans_ground,
        latency_sec=lat,
        cost_usd=cost,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cached=cached
    ))

    # --- Node 5: Utility Check (Self-RAG) ---
    util_prompt = format_registry_prompt("utility", prompt_version, question=question, generation=ans_gen)
    ans_util, lat, cost, cached, p_tok, c_tok = query_llm_with_metrics(util_prompt, api_client, use_cache)
    
    trace.add_span(Span(
        name="utility_check",
        input_data=util_prompt,
        output_data=ans_util,
        latency_sec=lat,
        cost_usd=cost,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        cached=cached
    ))

    # Compute aggregate metadata stats
    trace.metadata["total_latency_sec"] = sum(s.latency_sec for s in trace.spans)
    trace.metadata["total_cost_usd"] = sum(s.cost_usd for s in trace.spans)
    trace.metadata["total_prompt_tokens"] = sum(s.prompt_tokens for s in trace.spans)
    trace.metadata["total_completion_tokens"] = sum(s.completion_tokens for s in trace.spans)
    trace.metadata["status"] = "success"
    trace.metadata["working_context"] = working_context
    trace.metadata["answer"] = ans_gen
    
    return trace

# %% [markdown]
# ## 🧪 8. MLOps/LLMOps Tracking Integration
#
# We log prompt evaluation metrics, latency, and costs to MLflow, and upload the full structured `trace.json` file as a run artifact.

# %%
def log_llmops_run_to_mlflow(
    trace: Trace, 
    prompt_version: str, 
    faith_score: float, 
    recall_score: float, 
    judge_score: float
):
    """
    Logs prompt parameters, pipeline execution stats, and evaluation metrics into MLflow.
    Uploads the complete span-level trace JSON file as a run artifact.
    """
    trace_dict = trace.to_dict()
    
    # Enrich trace evaluations
    trace.evaluations = {
        "overlap_faithfulness": faith_score,
        "context_recall": recall_score,
        "llm_judge_grounding": judge_score
    }
    
    # Sync with run dictionary
    trace_dict["evaluations"] = trace.evaluations

    with mlflow.start_run(run_name=f"prompt_eval_{prompt_version}"):
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("model_name", trace.metadata.get("model_name", "unknown"))
        mlflow.log_param("trace_id", trace.trace_id)
        
        # Log aggregated metrics
        mlflow.log_metric("total_latency_sec", trace.metadata.get("total_latency_sec", 0.0))
        mlflow.log_metric("total_cost_usd", trace.metadata.get("total_cost_usd", 0.0))
        mlflow.log_metric("prompt_tokens", trace.metadata.get("total_prompt_tokens", 0))
        mlflow.log_metric("completion_tokens", trace.metadata.get("total_completion_tokens", 0))
        
        # Log quality metrics
        mlflow.log_metric("overlap_faithfulness", faith_score)
        mlflow.log_metric("context_recall", recall_score)
        mlflow.log_metric("llm_judge_grounding", judge_score)
        
        # Write structured trace tree to file and log as MLflow artifact
        artifact_path = "trace.json"
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(trace_dict, f, indent=4)
        mlflow.log_artifact(artifact_path)
        
        print(f"✅ LLMOps Run logged to MLflow under experiment 'LLMOps_Evaluation_Module_16'!")
        print(f"📁 Full structured span-level trace saved and uploaded as artifact: {artifact_path}")

# %% [markdown]
# ### Running the Simulation Trace & Logging to MLflow
#
# Let's execute the trace on two sequential queries, evaluate RAG quality, and persist metrics to MLflow:

# %%
if __name__ == "__main__":
    doc_context = "Special Operations Forces conduct missions globally. Mars is currently uninhabited."
    query = "Are there military underground bases on Mars?"
    gt = "No active bases on Mars."

    # Run 1: Cache Miss
    print("\n🟢 Run 1: Executing Agentic RAG Pipeline (Cache Miss)")
    trace_run = simulate_agentic_rag(query, doc_context, gt, "v2.0.0", client, use_cache=True)
    ans = trace_run.metadata["answer"]
    working_ctx = trace_run.metadata["working_context"]
    
    print("\n--- Telemetry Timeline (Spans) ---")
    for s in trace_run.spans:
        print(f"  [{s.name}] Latency: {s.latency_sec:.3f}s | Tokens: {s.prompt_tokens + s.completion_tokens} | Cost: ${s.cost_usd:.6f} | Cached: {s.cached}")
        
    print(f"\n  Trace ID:         {trace_run.trace_id}")
    print(f"  Total Latency:    {trace_run.metadata['total_latency_sec']:.3f}s")
    print(f"  Total Cost:       ${trace_run.metadata['total_cost_usd']:.6f}")
    print(f"  Answer:           {ans.strip()}")

    # RAG metrics evaluations
    faith_score = compute_faithfulness_overlap(ans, working_ctx)
    recall_score = compute_context_recall_overlap(gt, working_ctx)
    judge_score = llm_judge_grounding(working_ctx, ans, client)

    print(f"\n📊 RAG Quality Evaluations:")
    print(f"  Overlap Faithfulness: {faith_score * 100:.1f}%")
    print(f"  Context Recall:       {recall_score * 100:.1f}%")
    print(f"  LLM Judge Grounding:  {judge_score}/5.0")

    # Log evaluations to MLflow
    log_llmops_run_to_mlflow(trace_run, "v2.0.0", faith_score, recall_score, judge_score)
