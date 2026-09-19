import time
import pickle
import logging
from typing import Any
from prodml import config

logger = logging.getLogger("prodml")


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"Execution time for {func.__name__}: {end - start:.4f} seconds")
        return result

    return wrapper


class DurationPredictor:
    def __init__(self):
        self.model = None
        self.dv = None

    def load(self):
        with open(config.MODEL_PATH, "rb") as f:
            self.dv, self.model = pickle.load(f)

    @timed
    def predict_one(self, features_dict: dict[str, Any]) -> float:
        if self.model is None:
            self.load()
        X = self.dv.transform([features_dict])
        pred = self.model.predict(X)
        return float(pred[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        if self.model is None:
            self.load()
        X = self.dv.transform(features_list)
        preds = self.model.predict(X)
        return [float(p) for p in preds]
