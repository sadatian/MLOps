---
trigger: always_on
---

# Agent Rules for MLOps Tutorials Project

Whenever you are asked to make changes, debug, or write new tutorials in this workspace, you MUST adhere to the following rules:

## 1. Review Project Instructions First
- Before performing any code edits or file creation, you MUST read and review the [project-instruction.md](file:///home/t/MLOps/project-instruction.md) file.
- This file acts as the repository blueprint, containing structural roadmaps, todo lists, and styling standards. Do not deviate from the guidelines defined there.

## 2. Token-Efficiency & Resource Controls
- **NO Automatic Docs Build:** Do NOT run `mkdocs build` or `mkdocs serve` automatically. Doing so produces massive outputs and compiles many files, consuming excessive tokens and CPU.
- **Python Execution First:** Verify code correctness, syntax, and imports by executing python files directly using Python: `uv run python path/to/tutorial.py`.
- **User Prompts for Builds:** Only run documentation build processes when the user explicitly requests it or when a complete build is strictly required to verify rendering bugs. Prompt the user first.
- **Targeted File Reading:** When inspecting existing files, use specific ranges (`StartLine` / `EndLine`) in file-viewing tools instead of reading full 800-line blocks when only a few lines are needed.
- **Concise Reporting:** Keep communication and updates short, highlighting key outcomes, and avoiding long explanations of unchanged code.

## 3. Local LLM & Mock Routing
- **LLM Redirect:** If any LLM functionalities (evaluation, generation, prompt logging) are introduced, hardcode/configure them to route to `http://localhost:5055/v1` with a dummy API key. Do not make calls to external cloud LLM providers.
- **AWS Simulation:** If any AWS services (e.g., S3 storage for models, inputs, or data registry) are needed, mock them using `moto` (using mock decorators or standalone mock endpoints). Never require real AWS credentials.
- **Docker Simulation:** Explain container orchestration using standalone docker guides, Dockerfiles, and test queries locally.

## 4. Model Serialization & Standards
- **ONNX Serialization:** Models should be serialized with ONNX when possible.