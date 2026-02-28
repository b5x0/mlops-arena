"""
CIFAR-10 Training Pipeline using ZenML + MLflow + TensorFlow

Stack requirements:
  - Artifact Store: minio_store
  - Experiment Tracker: mlflow_tracker
  - Model Registry: mlflow_registry
  - Orchestrator: default
"""

import os
import csv
import logging
import botocore.endpoint

# ---------------------------------------------------------
# WINDOWS NETWORKING HOTFIX: Redirect host.docker.internal
# ---------------------------------------------------------
original_make_request = botocore.endpoint.Endpoint.make_request
def patched_make_request(self, operation_model, request_dict):
    if "request_dict" in locals() or "request_dict" in globals():
        url = request_dict.get("url", "")
        if "host.docker.internal" in url:
            request_dict["url"] = url.replace("host.docker.internal", "localhost")
    return original_make_request(self, operation_model, request_dict)
botocore.endpoint.Endpoint.make_request = patched_make_request
# ---------------------------------------------------------

import pandas as pd
import subprocess
from typing import Tuple

import mlflow
import mlflow.tensorflow
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers

from zenml import step, pipeline
from zenml.client import Client

def ensure_mlflow_env():
    """Ensure S3 credentials and endpoint are set for the current process."""
    from dotenv import load_dotenv
    load_dotenv()
    # Host-perspective endpoint
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "admin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password123"
    # boto3 needs to know to use path addressing instead of virtual routing for minio
    os.environ["S3_USE_PATH_STYLE_ENDPOINT"] = "true"
    os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"


# ---------------------------------------------------------------------------
# Step 1 - Data Ingestion
# ---------------------------------------------------------------------------

@step
def ingest_data() -> Tuple[np.ndarray, np.ndarray]:
    """Runs dvc pull and loads images/labels as NumPy arrays."""
    ensure_mlflow_env()
    print("==> [ingest_data] Running dvc pull ...")
    subprocess.run(["dvc", "pull"], capture_output=True)

    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    if not os.path.isdir(images_dir) or not os.path.isfile(metadata_path):
        raise FileNotFoundError("Data not found. Run download_data.py first.")

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    images, labels = [], []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(images_dir, row["filename"])
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(np.array(img, dtype=np.uint8))
                labels.append(class_to_idx[row["label"]])
            except Exception as e:
                print(f"  Warning: skipping {img_path}: {e}")

    return np.stack(images), np.array(labels, dtype=np.int32)

# ---------------------------------------------------------------------------
# Step 2 - Preprocessing
# ---------------------------------------------------------------------------

@step
def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes and splits data 80/20 with OHE labels."""
    ensure_mlflow_env()
    print("==> [preprocess_data] Scaling and splitting ...")
    X = X.astype("float32") / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)

# ---------------------------------------------------------------------------
# Step 3 - Model Training
# ---------------------------------------------------------------------------

@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    filters_1: int = 32,
    filters_2: int = 64,
    filters_3: int = 128
) -> keras.Model:
    """Trains a CNN with MLflow autologging."""
    ensure_mlflow_env()
    print("==> [train_model] Training CNN ...")
    mlflow.tensorflow.autolog()
    mlflow.log_params({
        "filters_1": filters_1,
        "filters_2": filters_2,
        "filters_3": filters_3,
        "dropout_rate": dropout_rate
    })

    # Data Augmentation layer as part of the Sequential model to boost accuracy
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        data_augmentation,
        # Block 1
        layers.Conv2D(filters_1, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        # Block 2
        layers.Conv2D(filters_2, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        # Block 3
        layers.Conv2D(filters_3, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    # Register the model in the SAME step where it was logged by autolog
    print("==> [train_model] Registering model as 'cifar10_classifier' ...")
    run_id = mlflow.active_run().info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "cifar10_classifier")
    
    return model

# ---------------------------------------------------------------------------
# Step 4 - Evaluation + Confusion Matrix
# ---------------------------------------------------------------------------

@step(experiment_tracker="mlflow_tracker")
def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluates and logs confusion matrix."""
    ensure_mlflow_env()
    print("==> [evaluate_model] Evaluating ...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"==> [evaluate_model] Logging artifact: {cm_path}")
    mlflow.log_artifact(cm_path)
    print("==> [evaluate_model] Complete.")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@pipeline(name="cifar10_training_pipeline")
def training_pipeline():
    X, y = ingest_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
