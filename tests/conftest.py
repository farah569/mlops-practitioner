import unittest
from unittest.mock import MagicMock, patch

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
    with patch("prodml.predict.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [15.5]
        mock_load.return_value = mock_model

        with (
            patch("builtins.open", unittest.mock.mock_open()),
            patch("pickle.load") as mock_pickle,
        ):
            mock_dv = MagicMock()
            mock_dv.transform.return_value.toarray.return_value = [[1, 2]]
            mock_pickle.return_value = (mock_dv, None)

            predictor = DurationPredictor()
            predictor.load()
            return predictor


@pytest.fixture
def sample_features():
    return {"PULocationID": 132, "DOLocationID": 264, "trip_distance": 5.0}
