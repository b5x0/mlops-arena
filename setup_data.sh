#!/bin/bash
set -e

echo "========================================="
echo "       DVC Data Setup Script             "
echo "========================================="

# Check for .env file to load variables
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found. Make sure it exists in the current directory."
  exit 1
fi

echo "1. Installing required Python dependencies..."
pip install "dvc[s3]" torchvision pillow

echo ""
echo "2. Running download_data.py to get CIFAR-10 data..."
python download_data.py

echo ""
echo "3. Initializing Git & DVC..."
if [ ! -d ".git" ]; then
  git init
fi
dvc init -f

echo ""
echo "4. Configuring DVC Remote for MinIO..."
# The bucket is mapped via the .env variable, using the default mlops-bucket
dvc remote add -d minio_remote s3://${MINIO_BUCKET}
dvc remote modify minio_remote endpointurl http://localhost:9000
dvc remote modify minio_remote access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify minio_remote secret_access_key ${AWS_SECRET_ACCESS_KEY}
dvc remote modify minio_remote use_ssl false

echo ""
echo "5. Adding the data folder to DVC tracking..."
dvc add data

echo ""
echo "6. Committing DVC configuration changes to Git..."
git add data.dvc .gitignore .dvc/config
git commit -m "Add CIFAR-10 dataset tracked by DVC" || echo "Nothing to commit"

echo ""
echo "7. Pushing tracked data to MinIO remote..."
dvc push

echo ""
echo "========================================="
echo "       Data Versioning Complete!         "
echo "========================================="
echo "The CIFAR-10 dataset has been pushed to your MinIO bucket."
