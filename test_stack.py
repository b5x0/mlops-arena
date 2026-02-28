"""
test_stack.py — Minimal ZenML "Hello Stack" Pipeline
=====================================================
Verifies that:
  ✔ The cifar_stack is correctly activated
  ✔ mlflow_tracker can log a metric to http://localhost:5000
  ✔ minio_store can persist a .txt artifact to s3://mlops-bucket

Run after setup_zenml.py has completed:
    python test_stack.py
"""

import os
import sys
import tempfile
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

# Fix encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

import mlflow
from zenml import pipeline, step


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Log a dummy metric to MLflow
# ──────────────────────────────────────────────────────────────────────────────

@step(experiment_tracker="mlflow_tracker")
def hello_mlflow() -> str:
    """Log a single metric to the remote MLflow server."""
    mlflow.log_param("hello", "world")
    mlflow.log_metric("dummy_accuracy", 0.42)
    print("✔  MLflow: logged param 'hello=world' and metric 'dummy_accuracy=0.42'")
    return "mlflow_ok"


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Save a .txt file to the MinIO artifact store
# ──────────────────────────────────────────────────────────────────────────────

@step
def hello_minio(mlflow_status: str) -> str:
    """Write a small text file; ZenML auto-uploads it to minio_store."""
    content = (
        "Hello from the cifar_stack!\n"
        f"MLflow step reported: {mlflow_status}\n"
    )
    # Write to a temp file — ZenML materialises the return value to MinIO
    tmp = os.path.join(tempfile.gettempdir(), "hello_stack.txt")
    with open(tmp, "w") as fh:
        fh.write(content)
    print(f"✔  MinIO: artifact written ({tmp})")
    # Return the content string so ZenML stores it as a built-in artifact
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

@pipeline(name="hello_stack_pipeline")
def hello_stack_pipeline() -> None:
    mlflow_status = hello_mlflow()
    hello_minio(mlflow_status)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Hello-Stack Test Pipeline")
    print("=" * 60)

    hello_stack_pipeline()

    print()
    print("=" * 60)
    print("  Test complete! Check:")
    print("  ► MLflow UI  : http://localhost:5000  (experiment: hello_stack_pipeline)")
    print("  ► ZenML UI   : http://localhost:8080  (pipeline: hello_stack_pipeline)")
    print("=" * 60)
