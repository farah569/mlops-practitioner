import pytest
from fastapi.testclient import TestClient
from prodml.api.main import app
from prodml.predict import DurationPredictor


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def trained_model():
    predictor = DurationPredictor()
    predictor.load()
    return predictor


@pytest.fixture
def sample_features():
    return {"PULocationID": 132, "DOLocationID": 264, "trip_distance": 5.0}
