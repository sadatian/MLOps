# %% [markdown]
# # 📝 Notebook Documentation with Jupytext & MkDocs
#
# This tutorial explains how Python scripts formatted with interactive percent-cells (`# %%`) serve as executable scripts AND documentation pages.
#
# ### The Jupytext/MkDocs Documentation Pipeline
#
# Static documentation websites often get out of sync with code. To avoid this, we use a toolchain that dynamically compiles executable, unit-tested code files directly into documentation notebooks.
#
# ```mermaid
#  graph TD
#      subgraph local_ide_development ["Local IDE (Development)"]
#          A["Developer edits src/module.py"] -->|"Percent cell syntax # %%"| B["Standard Python Script"]
#          B -->|"Interactive execution"| C["Fast loop & debugging"]
#      end

#      subgraph git_ci_cd ["Git & CI/CD"]
#          D["Git Version Control"]
#      end

#      subgraph documentation_generator_mkdocs_build ["Documentation Generator (mkdocs build)"]
#          E["Parse percent syntax"]
#          E -->|"Inject Jupytext Parser"| F["Convert to Jupyter ipynb format"]
#          F -->|"Execute cells if cache misses"| G["Capture stdout, tables & plots"]
#          G -->|"Markdown rendering with MathJax"| H["Static HTML Notebook File"]
#          H -->|"Theme integration"| I["Deployable site docs/site/"]
#      end

#      B -->|"Git Commit (Pure text diff)"| D
#      B -.->|mkdocs-jupyter plugin| E
#
#      C ~~~ D
#      C ~~~ E
#
# ```
#
# ## 💻 1. Interactive Percent Cells (`# %%`)
# By dividing our python scripts using `# %%` and `# %% [markdown]`, standard Python IDEs (such as VS Code, PyCharm, or JupyterLab) recognize them as Jupyter Notebook cells.
# This gives you the best of both worlds:
# - Version control friendly (they are pure text files with `.py` extension - no JSON metadata diff nightmare).
# - Executable cell-by-cell.
# - Renders into beautiful, readable notebooks.


# %%
# Ensure we run from the project root directory
import os
import sys

# Locate project root (searching upwards for mkdocs.yml)
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
while current_dir != os.path.dirname(current_dir):
    if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
        break
    current_dir = os.path.dirname(current_dir)

if os.path.exists(os.path.join(current_dir, "mkdocs.yml")):
    os.chdir(current_dir)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
else:
    print("⚠️ Could not find project root containing mkdocs.yml.")

# This is a standard python cell
message = "Hello from a standard Python cell!"
print(message)

# %% [markdown]
# ## 📝 2. Markdown Cells
# Markdown cells are specified using `# %% [markdown]`. Every subsequent line starting with `# ` is parsed as standard markdown.
# For example, we can render tables:
#
# | Feature | Jupyter Notebook (`.ipynb`) | Celled Python Script (`.py`) |
# |---|---|---|
# | Version Control | ❌ Diff is hard to read (large JSON) | ✅ Diff is clean (pure python) |
# | Execution | ✅ Cell-by-cell | ✅ Cell-by-cell |
# | Static Compilation | ✅ Supported | ✅ Supported (via `mkdocs-jupyter`) |
#
# We can also write code blocks within markdown or use LaTeX equations for math:
#
# $$E = mc^2$$

# %%
# Let's perform a simple math operation in python
x = 5
y = 10
result = x * y
print(f"The result of {x} * {y} is {result}")

# %% [markdown]
# ## ⚙️ 3. Compilation Config in `mkdocs.yml`
# The `mkdocs-jupyter` plugin has been configured to watch for `.py` files inside the `src/` directory and compile them automatically.
# Here is the relevant configuration in `mkdocs.yml`:
#
# ```yaml
# plugins:
#   - mkdocs-jupyter:
#       include_source: true
#       include: ["src/**/*.py"]
# ```
#
# ### 🖥️ 4. Integrating CLI commands for Docs
# Module 2 extends the CLI by adding `mlops docs`:
# * **Build the site documentation:**
#   ```bash
#   uv run mlops docs build
#   ```
# * **Serve the site documentation locally:**
#   ```bash
#   uv run mlops docs serve --host 127.0.0.1 --port 8000
#   ```
# * **Run via Docker:**
#   ```bash
#   docker run --rm -it -p 8000:8000 mlops-cli docs serve --host 0.0.0.0 --port 8000
#   ```
#
# Let's test checking the docs help menu:

# %%
import subprocess
result = subprocess.run(["mlops", "docs", "--help"], capture_output=True, text=True)
print(result.stdout)

# %% [markdown]
# Now that you understand how these scripts are compiled and managed via CLI, let's step into the AWS S3 Simulation guide to learn about AWS mocking, followed by the Data Version Control (DVC) guide!
