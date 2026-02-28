"""
Phase 2 Monitoring Pipeline using ZenML + Evidently
"""
import os
import logging
import botocore.endpoint

# ---------------------------------------------------------
# WINDOWS NETWORKING HOTFIX: Redirect host.docker.internal
# ---------------------------------------------------------
original_make_request = botocore.endpoint.Endpoint.make_request
def patched_make_request(self, operation_model, request_dict):
    url = request_dict.get("url", "")
    if "host.docker.internal" in url:
        request_dict["url"] = url.replace("host.docker.internal", "localhost")
    return original_make_request(self, operation_model, request_dict)
botocore.endpoint.Endpoint.make_request = patched_make_request
# ---------------------------------------------------------

import pandas as pd
from sklearn.datasets import fetch_california_housing
from zenml import pipeline, step
from evidently import Report
from evidently.presets import DataDriftPreset

@step
def get_reference_data() -> pd.DataFrame:
    """Load reference (baseline) dataset."""
    data = fetch_california_housing(as_frame=True).frame
    return data.sample(frac=0.5, random_state=0)

@step
def collect_inference_data() -> pd.DataFrame:
    """Load 'new' data to compare against reference (slightly different sample = simulated drift)."""
    data = fetch_california_housing(as_frame=True).frame
    return data.sample(frac=0.3, random_state=42)

@step
def run_evidently_report(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
    """Runs Evidently DataDriftPreset and returns whether drift was detected."""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data)

    # Save HTML report (Evidently 0.7.x)
    try:
        snapshot.save("evidently_arena_report.html")
        logging.info("Evidently report saved to evidently_arena_report.html")
    except Exception as e:
        logging.warning(f"Could not save HTML report: {e}")

    # Parse result from Evidently 0.7.x dict structure
    try:
        result_dict = report.as_dict()
        metrics = result_dict.get("metrics", [])
        # Look for drift summary in any metric result
        drift_detected = False
        for metric in metrics:
            result = metric.get("result", {})
            if "dataset_drift" in result:
                drift_detected = result["dataset_drift"]
                break
            if "drift_detected" in result:
                drift_detected = result["drift_detected"]
                break
    except Exception as e:
        logging.warning(f"Could not parse drift result: {e}. Defaulting to False.")
        drift_detected = False

    if drift_detected:
        logging.warning("⚠️ DATA DRIFT DETECTED: Retraining required!")
    else:
        logging.info("✅ No Data Drift. Model is stable.")

    return drift_detected

@pipeline(name="cifar10_monitoring_pipeline")
def monitoring_pipeline():
    """Phase 2 Pipeline for Evidently Data Drift Monitoring."""
    reference_data = get_reference_data()
    current_data = collect_inference_data()
    run_evidently_report(reference_data=reference_data, current_data=current_data)

if __name__ == "__main__":
    monitoring_pipeline()
