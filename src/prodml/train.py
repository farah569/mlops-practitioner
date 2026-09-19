import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from prodml import config
from prodml import data
from prodml import features
from prodml.logging_conf import setup_logging

logger = setup_logging()


def main():
    logger.info("Loading data...")
    df = data.load_data()

    logger.info("Cleaning and engineering features...")
    df = data.clean_data(df)
    df = features.create_features(df)

    logger.info("Preparing training data...")
    train_dicts = features.prepare_dictionaries(
        df, config.CATEGORICAL, config.NUMERICAL
    )

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df["duration"].values

    logger.info("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    logger.info("Evaluating...")
    y_pred = model.predict(X_train)
    rmse = mean_squared_error(y_train, y_pred) ** 0.5
    mae = mean_absolute_error(y_train, y_pred)

    logger.info(f"Validation RMSE: {rmse:.4f}")
    logger.info(f"Validation MAE: {mae:.4f}")

    logger.info("Saving model...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump((dv, model), f)
    logger.info(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
