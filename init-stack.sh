#!/bin/bash
echo "Starting the MLOps Stack with Docker Compose..."
docker-compose up -d

echo ""
echo "Waiting a few seconds for services to become healthy..."
sleep 15

echo ""
echo "=============================================="
echo "          MLOps Stack is up and running!      "
echo "=============================================="
echo "MinIO Console:    http://localhost:9001"
echo "MLflow UI:        http://localhost:5000"
echo "ZenML UI:         http://localhost:8080"
echo "=============================================="
echo "Use 'docker-compose ps' to check the status of your services."
