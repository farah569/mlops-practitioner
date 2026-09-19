# MLOps Mini Project 1: NYC Taxi Duration Predictor 

This is a production-ready Machine Learning service that predicts the duration of a taxi trip in New York City. The project has been refactored from a Jupyter notebook into a pip-installable Python package, exposed via a FastAPI service, and containerized using a multi-stage Docker build.

## Quickstart: Run in 3 Commands

You can pull the Docker image from Docker Hub and run the API directly without cloning the repository:

**1. Run the Docker container:**

docker run --rm -p 8000:8000 farah567/prodml-api:latest

**2. Check if the model is loaded (Health Check):**

curl http://localhost:8000/health


**3.Make a prediction:**

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "trip_distance": 5.0,
  "PU_DO": "132_264"
}'

📁 Repository Structure

mlops-practitioner/
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── 00-baseline.ipynb
├── src/
│   └── prodml/
│       ├── api/ (main.py, schemas.py)
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
├── tests/
└── reports/
    └── module-1.md
