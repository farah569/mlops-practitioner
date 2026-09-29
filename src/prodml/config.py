import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"

    class Config:
        env_file = ".env"
        extra = "ignore"


env_settings = Settings()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_URL = os.path.join(BASE_DIR, "data", "raw", "green_tripdata_2024-01.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "baseline.pkl")

CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]
