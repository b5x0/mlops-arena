import os
import sys
from dotenv import load_dotenv

# Force UTF-8 for Windows terminals to handle emojis (like MLflow's runner)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load credentials from .env
load_dotenv()

# Crucial: MLflow needs to know where MinIO is from the HOST perspective.
# Inside Docker it's http://minio:9000, but here it's http://localhost:9000.
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
os.environ["S3_USE_PATH_STYLE_ENDPOINT"] = "true"
# Also set these for ZenML/Boto3 fallback
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"

# Note: pipeline.py now contains a runtime monkeypatch to redirect 
# 'host.docker.internal' to 'localhost' automatically.

from pipeline import training_pipeline

if __name__ == "__main__":
    print("=" * 60)
    print("  CIFAR-10 Training Pipeline")
    print("=" * 60)

    # Run the pipeline
    training_pipeline()

    print()
    print("=" * 60)
    print("  Pipeline run complete!")
    print()
    print("  -> MLflow UI   : http://localhost:5000")
    print("  -> ZenML UI    : http://localhost:8080")
    print("  -> MinIO UI    : http://localhost:9001")
    print("=" * 60)
