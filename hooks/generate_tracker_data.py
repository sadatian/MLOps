import os
import json
import glob
import datetime

def altshell_formatter(source, language, css_class, options, md, **kwargs):
    import html
    import re
    cleaned_source = re.sub(r'^altshell\s*', '', source)
    escaped_source = html.escape(cleaned_source)
    return (
        f'<details class="admonition shell">'
        f'<summary class="admonition-title">Equivalent Shell Command</summary>'
        f'<pre><code class="language-bash">{escaped_source}</code></pre>'
        f'</details>'
    )

def on_config(config, **kwargs):
    superfences_config = config.setdefault("mdx_configs", {}).setdefault("pymdownx.superfences", {})
    custom_fences = superfences_config.setdefault("custom_fences", [])
    custom_fences.append({
        "name": "altshell",
        "class": "altshell",
        "format": altshell_formatter
    })
    return config

def on_post_build(config, **kwargs):
    print("Generating live tracker data...")
    data = {
        "data_version": "N/A",
        "data_last_updated": "N/A",
        "model_name": "N/A",
        "model_version": "N/A",
        "model_last_updated": "N/A",
        "prompt_version": "N/A",
        "latest_runs": []
    }
    
    # 1. Query DVC data version from data/*.dvc files
    try:
        dvc_files = glob.glob("data/**/*.dvc", recursive=True) + glob.glob("*.dvc")
        if dvc_files:
            latest_dvc = max(dvc_files, key=os.path.getmtime)
            mtime = os.path.getmtime(latest_dvc)
            data["data_last_updated"] = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            with open(latest_dvc, 'r') as f:
                lines = f.readlines()
                outs_section = False
                for line in lines:
                    if "outs:" in line:
                        outs_section = True
                    if outs_section and "md5:" in line:
                        data["data_version"] = line.split("md5:")[-1].strip()
                        break
    except Exception as e:
        print(f"Hook warning (DVC parsing): {e}")

    # 2. Query MLflow models and runs
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        registered_models = client.search_registered_models()
        if registered_models:
            # Sort models by last updated
            latest_model = max(registered_models, key=lambda m: m.last_updated_timestamp if m.last_updated_timestamp else 0)
            data["model_name"] = latest_model.name
            
            if latest_model.latest_versions:
                latest_ver = max(latest_model.latest_versions, key=lambda v: v.last_updated_timestamp if v.last_updated_timestamp else 0)
                data["model_version"] = f"v{latest_ver.version}"
                if latest_ver.last_updated_timestamp:
                    data["model_last_updated"] = datetime.datetime.fromtimestamp(latest_ver.last_updated_timestamp / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Hook warning (MLflow client query): {e}")
        
    try:
        import mlflow
        experiments = mlflow.search_experiments()
        if experiments:
            exp_ids = [e.experiment_id for e in experiments]
            runs_df = mlflow.search_runs(experiment_ids=exp_ids)
            if not runs_df.empty:
                # Sort runs by start time descending
                if 'start_time' in runs_df.columns:
                    runs_df = runs_df.sort_values(by='start_time', ascending=False)
                
                # Try to find the latest prompt version from any run
                prompt_ver = "N/A"
                for col in runs_df.columns:
                    if col == "params.prompt_version":
                        valid_versions = runs_df[runs_df[col].notna()][col]
                        valid_versions = valid_versions[valid_versions.astype(str).str.strip() != ""]
                        valid_versions = valid_versions[valid_versions.astype(str).str.strip().str.upper() != "N/A"]
                        if not valid_versions.empty:
                            prompt_ver = str(valid_versions.iloc[0])
                            break
                data["prompt_version"] = prompt_ver

                # Get the most recent run
                latest_run = runs_df.iloc[0]
                run_info = {
                    "experiment_name": next((e.name for e in experiments if e.experiment_id == latest_run["experiment_id"]), "Unknown"),
                    "run_id": latest_run["run_id"][:8],
                    "status": latest_run["status"],
                    "metrics": {},
                    "params": {}
                }
                for col in runs_df.columns:
                    if col.startswith("metrics."):
                        metric_name = col.replace("metrics.", "")
                        val = latest_run[col]
                        if val is not None and not (isinstance(val, float) and (val != val)): # Check NaN
                            run_info["metrics"][metric_name] = round(float(val), 4)
                    elif col.startswith("params."):
                        param_name = col.replace("params.", "")
                        val = latest_run[col]
                        if val is not None and str(val) != "nan" and str(val).strip() != "":
                            run_info["params"][param_name] = str(val)
                data["latest_runs"].append(run_info)
    except Exception as e:
        print(f"Hook warning (MLflow runs query): {e}")

    # Write to site_dir/tracker_status.json to prevent infinite livereload loops
    output_path = os.path.join(config["site_dir"], "tracker_status.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Live tracker data successfully written to {output_path}")

def on_post_page(output, page, config, **kwargs):
    import re
    # 1. Replace the raw HTML pre class from "mermaid" to "jp-Mermaid-code"
    # to prevent MkDocs Material's theme loader from double-rendering it.
    output = re.sub(
        r'<div class="jp-Mermaid">\s*<pre class="mermaid">',
        '<div class="jp-Mermaid"><pre class="jp-Mermaid-code">',
        output
    )
    
    # 2. Update the inline script query selector to target the new class name
    output = output.replace(
        '.querySelectorAll(".jp-Mermaid > pre.mermaid")',
        '.querySelectorAll(".jp-Mermaid > pre.jp-Mermaid-code")'
    )
    
    # 3. Replace the dynamic import block with window.mermaid
    pattern = (
        r'const\s+mermaid\s*=\s*\(await\s+import\("https://cdnjs\.cloudflare\.com/ajax/libs/mermaid/11\.10\.0/mermaid\.esm\.min\.mjs"\)\)\.default;'
        r'\s*const\s+elkUrl\s*=\s*"https://cdnjs\.cloudflare\.com/ajax/libs/mermaid-layout-elk/0\.1\.9/mermaid-layout-elk\.esm\.min\.mjs";'
        r'\s*if\s*\(elkUrl\)\s*{\s*const\s+elkLayouts\s*=\s*\(await\s+import\(elkUrl\)\)\.default;\s*mermaid\.registerLayoutLoaders\(elkLayouts\);\s*}'
    )
    
    replacement = (
        'const mermaid = window.mermaid;\n'
        '    if (!mermaid) {\n'
        '      console.warn("window.mermaid is not defined");\n'
        '      return;\n'
        '    }'
    )
    
    output = re.sub(pattern, replacement, output)

    # 4. Inject customized Mermaid theme settings and fonts matching the site palette
    init_pattern = (
        r'mermaid\.initialize\(\{\s*'
        r'maxTextSize:\s*100000,\s*'
        r'maxEdges:\s*100000,\s*'
        r'startOnLoad:\s*false,\s*'
        r'fontFamily:\s*window\s*\.getComputedStyle\(document\.body\)\s*\.getPropertyValue\("--jp-ui-font-family"\),\s*'
        r'theme:\s*document\.querySelector\("body\[data-jp-theme-light=\'true\'\]"\)\s*\?\s*"default"\s*:\s*"dark",\s*'
        r'\}\);'
    )
    
    init_replacement = (
        'const isDark = document.body.getAttribute("data-md-color-scheme") === "slate";\n'
        '    mermaid.initialize({\n'
        '      maxTextSize: 100000,\n'
        '      maxEdges: 100000,\n'
        '      startOnLoad: false,\n'
        '      fontFamily: "Nunito, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif",\n'
        '      theme: "base",\n'
        '      themeVariables: {\n'
        '        fontSize: "13px",\n'
        '        background: isDark ? "#2e3035" : "#ffffff",\n'
        '        primaryColor: isDark ? "#4a2339" : "#f8f1f3",\n'
        '        primaryTextColor: isDark ? "#e4e4e7" : "#3f4650",\n'
        '        primaryBorderColor: "#7d2a44",\n'
        '        lineColor: isDark ? "#a6536b" : "#7d2a44",\n'
        '        secondaryColor: isDark ? "#1a1a1e" : "#fdfdfd",\n'
        '        tertiaryColor: isDark ? "#25252b" : "#fafafa",\n'
        '        actorBkg: isDark ? "#4a2339" : "#f8f1f3",\n'
        '        actorBorder: "#7d2a44",\n'
        '        actorTextColor: isDark ? "#e4e4e7" : "#3f4650",\n'
        '        signalColor: isDark ? "#bc7085" : "#7d2a44",\n'
        '        signalTextColor: isDark ? "#bc7085" : "#7d2a44",\n'
        '        labelBoxBkgColor: isDark ? "#4a2339" : "#f8f1f3",\n'
        '        labelBoxBorderColor: "#7d2a44",\n'
        '        labelTextColor: isDark ? "#e4e4e7" : "#3f4650",\n'
        '        loopBkgColor: isDark ? "#25252b" : "#fafafa",\n'
        '        noteBkgColor: isDark ? "#25252b" : "#fcf8e3",\n'
        '        noteBorderColor: isDark ? "#a6536b" : "#faebcc",\n'
        '        noteTextColor: isDark ? "#e4e4e7" : "#8a6d3b"\n'
        '      }\n'
        '    });'
    )

    output = re.sub(init_pattern, init_replacement, output)

    # 5. Convert any raw altshell code blocks that bypassed superfences (e.g., from mkdocs-jupyter conversion)
    def replace_altshell(match):
        code_content = match.group(1)
        import re
        cleaned_content = re.sub(r'^altshell\s*', '', code_content)
        return (
            f'<details class="admonition shell">'
            f'<summary class="admonition-title">Equivalent Shell Command</summary>'
            f'<pre><code class="language-bash">{cleaned_content}</code></pre>'
            f'</details>'
        )
    output = re.sub(r'<div class="highlight"><pre><code class="language-altshell">([\s\S]*?)</code></pre></div>', replace_altshell, output)
    output = re.sub(r'<pre><code class="language-altshell">([\s\S]*?)</code></pre>', replace_altshell, output)

    return output

