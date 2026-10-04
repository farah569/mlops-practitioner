import os
import pickle
import time

import matplotlib.pyplot as plt
import mlflow
import optuna
import torch
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from prodml import config, data, features
from prodml.logging_conf import setup_logging

logger = setup_logging()

mlflow.set_tracking_uri(config.env_settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment("NYC_Taxi_Duration")

logger.info("Preparing data...")
df = data.load_data()
df = data.clean_data(df)
df = features.create_features(df)
df = df.sample(n=5000, random_state=42)

train_dicts = features.prepare_dictionaries(df, config.CATEGORICAL, config.NUMERICAL)
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df["duration"].values

os.makedirs(config.MODEL_DIR, exist_ok=True)
with open(config.MODEL_PATH, "wb") as f:
    pickle.dump((dv, None), f)

COMMON_TAGS = {"author": "Farah", "data_version": "2024-01-raw", "git_commit": "HEAD"}


def train_linear_regression():
    with mlflow.start_run(run_name="Linear_Regression"):
        mlflow.set_tags(COMMON_TAGS)
        mlflow.set_tag("framework", "scikit-learn")

        start = time.time()
        model = LinearRegression()
        model.fit(X_train, y_train)
        duration = time.time() - start

        preds = model.predict(X_train)
        mae = mean_absolute_error(y_train, preds)
        rmse = mean_squared_error(y_train, preds) ** 0.5

        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "train_duration_sec": duration})

        plt.scatter(y_train, preds, alpha=0.5)
        plt.savefig("residual.png")
        mlflow.log_artifact("residual.png")

        mlflow.sklearn.log_model(model, "model")
        logger.info(f"Linear Regression MAE: {mae:.2f}")


def train_pytorch():
    with mlflow.start_run(run_name="PyTorch_MLP"):
        mlflow.set_tags(COMMON_TAGS)
        mlflow.set_tag("framework", "pytorch")

        X_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        model = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 10), nn.ReLU(), nn.Linear(10, 1)
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        start = time.time()
        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()
        duration = time.time() - start

        preds = model(X_tensor).detach().numpy()
        mae = mean_absolute_error(y_train, preds)

        mlflow.log_params({"epochs": 50, "lr": 0.1})
        mlflow.log_metrics({"MAE": mae, "train_duration_sec": duration})
        mlflow.pytorch.log_model(model, "model", input_example=X_tensor.numpy())
        logger.info(f"PyTorch MAE: {mae:.2f}")


def optimize_xgboost():
    mlflow.xgboost.autolog()

    def objective(trial):
        with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}", nested=True):
            mlflow.set_tags(COMMON_TAGS)
            mlflow.set_tag("framework", "xgboost")

            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "n_estimators": trial.suggest_int("n_estimators", 10, 50),
            }

            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_train)
            mae = mean_absolute_error(y_train, preds)
            return mae

    logger.info("Starting XGBoost Hyperparameter Sweep (10 runs)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    logger.info(f"Best XGBoost MAE: {study.best_value:.2f}")


def main():
    train_linear_regression()
    train_pytorch()
    optimize_xgboost()


if __name__ == "__main__":
    main()
