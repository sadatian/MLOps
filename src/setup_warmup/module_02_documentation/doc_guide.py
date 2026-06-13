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
# graph TD
#     subgraph Local IDE (Development)
#         A[Developer edits src/module.py] -->|Percent cell syntax # %%| B[Standard Python Script]
#         B -->|Interactive execution| C[Fast loop & debugging]
#     end
# 
#     subgraph Git & CI/CD
#         B -->|Git Commit (Pure text diff)| D[Git Version Control]
#     end
# 
#     subgraph Documentation Generator (mkdocs build)
#         B -.->|mkdocs-jupyter plugin| E[Parse percent syntax]
#         E -->|Inject Jupytext Parser| F[Convert to Jupyter ipynb format]
#         F -->|Execute cells if cache misses| G[Capture stdout, tables & plots]
#         G -->|Markdown rendering with MathJax| H[Static HTML Notebook File]
#         H -->|Theme integration| I[Deployable site docs/site/]
#     end
# ```
#
# ## 💻 1. Interactive Percent Cells (`# %%`)
# By dividing our python scripts using `# %%` and `# %% [markdown]`, standard Python IDEs (such as VS Code, PyCharm, or JupyterLab) recognize them as Jupyter Notebook cells.
# This gives you the best of both worlds:
# - Version control friendly (they are pure text files with `.py` extension - no JSON metadata diff nightmare).
# - Executable cell-by-cell.
# - Renders into beautiful, readable notebooks.


# %%
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
# Now that you understand how these scripts are compiled, let's step into the AWS S3 Simulation guide to learn about AWS mocking, followed by the Data Version Control (DVC) guide!
