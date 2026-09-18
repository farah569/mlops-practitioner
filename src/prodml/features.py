import pandas as pd
from typing import Any


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # merging drop off and pick times
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


def prepare_dictionaries(
    df: pd.DataFrame, categorical: list[str], numerical: list[str]
) -> list[dict[str, Any]]:
    # turning data to dictionary
    return df[categorical + numerical].to_dict(orient="records")
