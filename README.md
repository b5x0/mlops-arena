# CIFAR-10 MLOps Challenge

An end-to-end MLOps pipeline for CIFAR-10 image classification using **ZenML**, **MLflow**, **MinIO**, and **DVC**.

## Features
- **Infrastructure**: Containerized MLflow (tracking + registry), MinIO (artifact store), and ZenML Server.
- **Data Versioning**: DVC integration for tracking CIFAR-10 datasets.
- **Pipeline**: ZenML pipeline with steps for ingestion, preprocessing, training (CNN), and evaluation.
- **Tracking**: Automated logging of parameters, metrics (accuracy/loss), and artifacts (Confusion Matrix) to MLflow.
- **Model Registry**: Automated registration of trained models in the MLflow Model Registry.

---

## Prerequisites

- **Python 3.11**
- **Docker & Docker Compose**
- **Git**

### Dependencies
Install the required Python packages:
```powershell
pip install zenml mlflow tensorflow matplotlib seaborn scikit-learn pillow python-dotenv torchvision dvc[s3]
```

---

## Getting Started

### 1. Launch the Infrastructure
Start the Dockerized services (Postgres, MinIO, MLflow, ZenML):
```powershell
docker-compose up -d
```
*Wait ~30 seconds for services to initialize.*

### 2. Verify Health
Check if all endpoints are reachable:
```powershell
python check_infra.py
```

### 3. Setup ZenML Stack
Connect to the ZenML server and register the stack components (Flavor: S3 for MinIO, Flavor: MLflow for Tracking/Registry):
```powershell
python setup_zenml.py --reset
```
*Note: This script uses `admin` / `Password123.` by default from your `.env`.*

### 4. Prepare Data
Download the CIFAR-10 dataset and prepare it for DVC:
```powershell
python download_data.py
```

### 5. Run the Pipeline
Execute the full training and tracking flow:
```powershell
python run_pipeline.py
```

---

## Monitoring UIs

- **MLflow**: [http://localhost:5000](http://localhost:5000) (Metrics, Artifacts, Model Registry)
- **ZenML Dashboard**: [http://localhost:8080](http://localhost:8080) (Pipeline visualization)
- **MinIO Console**: [http://localhost:9001](http://localhost:9001) (Artifact storage)

---

## Project Structure
- `pipeline.py`: ZenML pipeline definition and steps.
- `setup_zenml.py`: Robust SDK-based stack registration.
- `check_infra.py`: Infrastructure health checker.
- `download_data.py`: CIFAR-10 downloader/processor.
- `run_pipeline.py`: Main entry point.
- `docker-compose.yml`: Definition of the MLOps backbone.