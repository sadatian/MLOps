import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

def test_evidently_drift_preset(tmp_path):
    # Setup small reference and current dataframes
    np.random.seed(42)
    reference = pd.DataFrame({
        "area_sqft": np.random.normal(1800, 300, size=20),
        "bedrooms": np.random.choice([2, 3, 4], size=20),
        "price_usd": np.random.normal(300000, 50000, size=20)
    })
    current = pd.DataFrame({
        "area_sqft": np.random.normal(2300, 400, size=20),
        "bedrooms": np.random.choice([2, 3, 4], size=20),
        "price_usd": np.random.normal(420000, 70000, size=20)
    })

    # Run evidently report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    
    html_output = tmp_path / "drift_report.html"
    snapshot.save_html(str(html_output))
    
    assert html_output.exists()
