from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.predict import DurationPredictor


@pytest.fixture(autouse=True)
def mock_predictor_for_api(monkeypatch):
    def fake_load(self):
        self.dv = MagicMock()
        self.dv.transform.return_value.toarray.return_value = [[1.0, 2.0]]
        self.model = MagicMock()
        self.model.predict.return_value = [19.5]

    monkeypatch.setattr(DurationPredictor, "load", fake_load)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def trained_model():
    predictor = DurationPredictor()
    predictor.load = MagicMock()
    predictor.dv = MagicMock()
    predictor.dv.transform.return_value.toarray.return_value = [[1.0, 2.0]]
    predictor.model = MagicMock()
    predictor.model.predict.return_value = [19.5]
    return predictor


@pytest.fixture
def sample_features():
    return {"PULocationID": 132, "DOLocationID": 264, "trip_distance": 5.0}
