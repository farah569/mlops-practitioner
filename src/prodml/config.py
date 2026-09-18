import os

# project path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data path
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "baseline.pkl")

# columns
CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]
