import os
import json
import glob
import datetime

def on_pre_build(config, **kwargs):
    print("Generating live tracker data...")
    data = {
        "data_version": "N/A",
        "data_last_updated": "N/A",
        "model_name": "N/A",
        "model_version": "N/A",
        "model_last_updated": "N/A",
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
                # Get the most recent run
                latest_run = runs_df.iloc[0]
                run_info = {
                    "experiment_name": next((e.name for e in experiments if e.experiment_id == latest_run["experiment_id"]), "Unknown"),
                    "run_id": latest_run["run_id"][:8],
                    "status": latest_run["status"],
                    "metrics": {}
                }
                for col in runs_df.columns:
                    if col.startswith("metrics."):
                        metric_name = col.replace("metrics.", "")
                        val = latest_run[col]
                        if val is not None and not (isinstance(val, float) and (val != val)): # Check NaN
                            run_info["metrics"][metric_name] = round(float(val), 4)
                data["latest_runs"].append(run_info)
    except Exception as e:
        print(f"Hook warning (MLflow runs query): {e}")

    # Write to docs/tracker_status.json
    output_path = os.path.join(config["docs_dir"], "tracker_status.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Live tracker data successfully written to {output_path}")
