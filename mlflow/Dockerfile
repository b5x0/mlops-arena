FROM python:3.9-slim

# Install MLflow and its dependencies for Postgres and MinIO (S3)
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

# Expose MLflow port
EXPOSE 5000
